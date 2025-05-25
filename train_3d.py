import os
import sys
import argparse
from pathlib import Path
import shutil

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.tuner import Tuner
import wandb

from perfusionforecast import config_map_3d
from perfusionforecast.models_3d import model_map
from perfusionforecast.utils import seed_everything, VolumeDataModule3D
from perfusionforecast.models_3d import Pl_Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory where the data is stored")
    parser.add_argument("--model_name", type=str, choices=model_map.keys(), help="Name of the model to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device the model runs on.")
    parser.add_argument("--gpu", type=int, default=1, help="Number of gpu the model runs on")
    parser.add_argument("--input_frames", type=int, default=9, help="Number of time-frames the model gets as input")
    parser.add_argument("--pred_frames", type=int, default=9, help="Number of time-frames the model needs to predict")
    parser.add_argument("--pred_n_frames_per_step", type=int, default=9, help="Number of time-frames that are predicted per step")
    parser.add_argument("--seed", type=int, default=100, help="Seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloaders")
    parser.add_argument("--epochs", type=int, default=40, help="Number of Epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning-Rate")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch Size")
    parser.add_argument("--precision", type=str, default="32-true", help="Precision")
    args = parser.parse_args()

    wandb.login()
    seed_everything(args.seed)
    
    input_shape = [args.input_frames, 1, 16, 256, 256]
    if args.model_name in config_map_3d:
        config_fn = config_map_3d[args.model_name]
        config = config_fn(in_shape=input_shape, pred_n_frames_per_step=args.pred_n_frames_per_step)
        print(config)
        model_fn = model_map[args.model_name]
        model = model_fn(**config)
    else:
        raise ValueError(f"Model '{args.model_name}' is not a valid option. Available: {list(config_map_3d.keys())}")

    run_name = args.model_name
    run_name = "3D-" + run_name
    run_name += f"_{args.pred_frames}"
    if args.pred_frames == args.pred_n_frames_per_step:
        run_name += "_NAR"
    elif args.pred_n_frames_per_step == 1:
        run_name += "_FAR"
    else:
        run_name += f"_PAR_{args.pred_n_frames_per_step}"

    # Get data module
    dm = VolumeDataModule3D(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
        drop_last= False,
        sequence_length=args.input_frames,
        prediction_length=args.pred_frames,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )
    
    wandb_logger = WandbLogger(entity="ChadCTP", project="perfusion-ct-prediction", name=run_name)
    
    added_configs = vars(args) | config
    added_configs["run_name"] = run_name
    pl_model = Pl_Model(
        passed_model=model,
        config=added_configs,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",  
        mode="min",  
        save_top_k=1,  
        filename="best-checkpoint",
        verbose=True,
    )
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices= [args.gpu] if args.device == "cuda" else None,
        max_epochs=args.epochs,
        callbacks=[RichProgressBar(), checkpoint_callback],
        check_val_every_n_epoch=5,
        precision=args.precision
    )

    print('>'*35 + ' Training ' + '<'*35)
    trainer.fit(
        model=pl_model,
        datamodule=dm,
    )
    
    #check and log the losses "to beat"
    print('>'*35 + ' Check Losses ' + '<'*35)
    dm.setup()
    pl_model.check_losses(dm.train_dataloader(), mode="train", use_wandb=True)
    pl_model.check_losses(dm.val_dataloader()[0], mode="val", use_wandb=True)
    pl_model.check_losses(dm.test_dataloader(), mode="test", use_wandb=True)
    
    #val_results = trainer.validate(pl_model, datamodule=dm)
    print('>'*35 + ' Testing ' + '<'*35)
    best_ckpt_path = Path(checkpoint_callback.best_model_path)
    target_dir = Path("./ModelWeights")
    target_dir.mkdir(parents=True, exist_ok=True)
    new_filename = f"{run_name}.ckpt"
    target_path = target_dir / new_filename
    shutil.copy(best_ckpt_path, target_path)

    pl_model = Pl_Model.load_from_checkpoint(
        target_path,
        passed_model=model,
    )
    test_results = trainer.test(pl_model, datamodule=dm)

    print('>'*35 + ' Processing Outputs ' + '<'*35)
    pl_model = Pl_Model.load_from_checkpoint(
        target_path,
        passed_model=model,
    )
    dst_outputs=Path(f"./outputs/{run_name}")
    dst_targets=Path(f"./targets/{run_name}")
    
    dst_outputs.mkdir(parents=True, exist_ok=True)
    dst_targets.mkdir(parents=True, exist_ok=True)
    
    dm.setup()
    if dm.test_paths != None:
        for path in dm.test_paths:
            output, target = pl_model.predict_one_ct(
                path, 
                input_frames=args.input_frames, 
                pred_frames=args.pred_frames,
                pred_n_frames_per_step=args.pred_n_frames_per_step,
            )

            dst_output_path=dst_outputs / Path(path).name
            output = output.cpu()
            np.save(dst_output_path, output.cpu())

            dst_target_path = dst_targets / Path(path).name
            target = target.cpu()
            np.save(dst_target_path, target)
