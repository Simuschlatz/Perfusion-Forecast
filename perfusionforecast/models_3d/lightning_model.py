from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

import wandb
import numpy as np

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from perfusionforecast.utils import HuberSSIMLoss2D, HuberSSIMLoss3D


class Pl_Model(pl.LightningModule):
    def __init__(
        self,
        passed_model: nn.Module,
        config: Dict[str, Any],
    ):
        super(Pl_Model, self).__init__()
        self.passed_model = passed_model
        self.config = config
        # ------------------------------------------------------------------------------
        # DELTA FACTOR (Scales how much delta is updated per epoch)
        # - Higher delta_factor → More aggressive updates to delta
        # - Lower delta_factor  → Smoother, slower changes to delta
        # ------------------------------------------------------------------------------
        # Recommended tuning:
        # - > 1.5  → If delta is too unstable (jumps too much)
        # - 1.2 - 1.5  → Best for gradual adaptation (Default)
        # - < 1.2  → If delta changes too slowly (use for very stable datasets)
        # ------------------------------------------------------------------------------
        # Default: delta_factor = 1.2 (Balanced adaptation)
        # ------------------------------------------------------------------------------
        self.delta_factor = 1.2  # More factor = Faster delta adjustments
        self.delta = 0.05
        self.previous_delta = self.delta
        
        #speicher alle parameter ab
        self.save_hyperparameters(ignore=["passed_model"])

        # Setup training components
        self.mae_criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()
        self.psnr_criterion = PeakSignalNoiseRatio()
        self.huberssim3d_criterion = HuberSSIMLoss3D()
        self.huber_criterion = nn.HuberLoss()
        

    def forward(self, x):
        x = self.passed_model(x)
        #Tanh has a larger gradient range, reducing saturation issues compared to sigmoid.
        #Allows more stable gradient flow for deep networks.
        x = 0.5*(F.tanh(x)+1)
        return x

    def configure_optimizers(self):
        """Sets the Optimizer for the Model"""
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.config['learning_rate'],
        )
        return [optimizer]

    def _calculate_loss(self, batch, mode="train"):
        """Calculates the loss for a batch in different modes (training, validation, testing)"""
        inputs, targets = batch

        #forward pass
        mae_loss = 0.0
        mse_loss = 0.0
        huber_loss = 0.0
        rmse_loss = 0.0
        ssim_loss = 0.0
        huberssim_loss = 0.0
        temporal_loss = 0.0
        psnr_loss = 0.0
        total_loss = 0.0
        for t in range(0, self.config["pred_frames"], self.config["pred_n_frames_per_step"]):
            if self.config["pred_frames"]-t<self.config["pred_n_frames_per_step"]:
                frames_this_step = self.config["pred_frames"]-t
            else:
                frames_this_step = self.config["pred_n_frames_per_step"]
            outputs = self.forward(inputs)
            #print(f"{t}:{t+frames_this_step}")
            #get only the first predicted frame
            outputs = outputs[:, :frames_this_step, :, :, :, :]

            #calcualte losses
            mae_loss_ = self.mae_criterion(outputs, targets[:, t:t+frames_this_step, :, :, :, :])
            mse_loss_ = self.mse_criterion(outputs, targets[:, t:t+frames_this_step, :, :, :, :])
            rmse_loss_ = torch.sqrt(self.mse_criterion(outputs, targets[:, t:t+frames_this_step, :, :, :, :]))
            psnr_loss_ = self.psnr_criterion(outputs, targets[:, t:t+frames_this_step, :, :, :, :])
            huberssim_loss_, huber_loss_, ssim_loss_, temporal_loss_ = self.huberssim3d_criterion(outputs, targets[:, t:t+frames_this_step, :, :, :, :])
            total_loss_ = huberssim_loss_ #self.huber_criterion(outputs, targets[:, t:t+frames_this_step, :, :])  

            mae_loss += mae_loss_
            mse_loss += mse_loss_
            huber_loss += huber_loss_
            rmse_loss += rmse_loss_
            ssim_loss += ssim_loss_
            huberssim_loss += huberssim_loss_
            temporal_loss += temporal_loss_
            psnr_loss += psnr_loss_
            total_loss += total_loss_
            
            inputs = torch.cat([inputs[:, self.config["pred_n_frames_per_step"]:, :, :, :, :], outputs], dim=1)

        #logging
        self.log(f"{mode}_mae_loss", mae_loss)
        self.log(f"{mode}_mse_loss", mse_loss)
        self.log(f"{mode}_huber_loss", huber_loss)
        self.log(f"{mode}_rmse_loss", rmse_loss)
        self.log(f"{mode}_ssim_loss", ssim_loss)
        self.log(f"{mode}_huberssim_loss", huberssim_loss)
        self.log(f"{mode}_temporal_loss", temporal_loss)
        self.log(f"{mode}_psnr_loss", psnr_loss)
        self.log(f"{mode}_total_loss", total_loss, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

        inputs, targets = batch
        outputs = []
        for t in range(0, self.config["pred_frames"], self.config["pred_n_frames_per_step"]):
            if self.config["pred_frames"]-t<self.config["pred_n_frames_per_step"]:
                frames_this_step = self.config["pred_frames"]-t
            else:
                frames_this_step = self.config["pred_n_frames_per_step"]
            outputs_t = self.forward(inputs)
            #print(f"{t}:{t+frames_this_step}")
            #get only the first predicted frame
            outputs_t = outputs_t[:, :frames_this_step, :, :, :, :]
            
            outputs.append(outputs_t)

            inputs = torch.cat([inputs[:, self.config["pred_n_frames_per_step"]:, :, :, :, :], outputs_t], dim=1)
            
            #concat time and add to overall lst
        outputs = torch.concat(outputs, dim=1)
        
        #calculate losses
        mae_loss = self.mae_criterion(outputs, targets)
        mse_loss = self.mse_criterion(outputs, targets)
        rmse_loss = torch.sqrt(self.mse_criterion(outputs, targets))
        huberssim_loss, huber_loss, ssim_loss, temporal_loss = self.huberssim3d_criterion(outputs, targets)
        psnr_loss = self.psnr_criterion(outputs, targets)
        total_loss = huberssim_loss

        #logging
        self.log(f"overall_val_mae_loss", mae_loss)
        self.log(f"overall_val_mse_loss", mse_loss)
        self.log(f"overall_val_huber_loss", huber_loss)
        self.log(f"overall_val_rmse_loss", rmse_loss)
        self.log(f"overall_val_ssim_loss", ssim_loss)
        self.log(f"overall_val_huberssim_loss", huberssim_loss)
        self.log(f"overall_val_temporal_loss", temporal_loss)
        self.log(f"overall_val_psnr_loss", psnr_loss)
        self.log(f"overall_val_total_loss", total_loss)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")

    def on_train_epoch_end(self):
        #adjust delta
        val_loader = self.trainer.datamodule.val_dataloader()
        all_errors = []
    
        with torch.no_grad():  
            for batch in val_loader:
                x, y = batch
                y_pred = self(x.to(self.device))
                error = torch.abs(y.to(self.device) - y_pred)
                all_errors.append(error.view(-1))
    
        all_errors = torch.cat(all_errors)
        new_delta = self.delta_factor * torch.std(all_errors).item()

        #Blend previous and new delta for smoother updates
        #is capped between 0.02 and 0.35 so that is the data is too noisy huber does not just become mse
        new_delta = min(0.5, max(0.02, 0.8 * self.previous_delta + 0.2 * new_delta))
        self.previous_delta = new_delta

        #update
        self.huberssim3d_criterion.delta = new_delta
    
        #logging
        self.log("delta", new_delta)
        
    @torch.no_grad()
    def check_losses(self, loader, mode, use_wandb=False):
        mae_loss = 0.0
        mse_loss = 0.0
        huber_loss = 0.0
        rmse_loss = 0.0
        ssim_loss = 0.0
        huberssim_loss = 0.0
        temporal_loss = 0.0
        psnr_loss = 0.0
        total_loss = 0.0
        for inputs, targets in loader:
            for t in range(self.config["pred_frames"]):
                mae_loss_ = self.mae_criterion(inputs[:, -1, :, :, :].unsqueeze(1), targets[:, t, :, :].unsqueeze(1))
                mse_loss_ = self.mse_criterion(inputs[:, -1, :, :, :].unsqueeze(1), targets[:, t, :, :].unsqueeze(1))
                huberssim_loss_, huber_loss_, ssim_loss_, temporal_loss_ = self.huberssim3d_criterion(inputs[:, -1, :, :, :].unsqueeze(1), targets[:, t, :, :].unsqueeze(1))
                rmse_loss_ = torch.sqrt(self.mse_criterion(inputs[:, -1, :, :, :].unsqueeze(1), targets[:, t, :, :].unsqueeze(1)))
                psnr_loss_ = self.psnr_criterion(inputs[:, -1, :, :, :].unsqueeze(1), targets[:, t, :, :].unsqueeze(1))
                total_loss_ = huberssim_loss_   
                
                mae_loss += mae_loss_.item()                
                mse_loss += mse_loss_.item()
                huber_loss += huber_loss_.item()
                rmse_loss += rmse_loss_.item()
                ssim_loss += ssim_loss_.item()
                huberssim_loss += huberssim_loss_.item()
                temporal_loss += temporal_loss_.item()
                psnr_loss += psnr_loss_.item()
                total_loss += total_loss_.item()
                
        mae_loss = mae_loss / len(loader)
        mse_loss = mse_loss / len(loader)
        huber_loss = huber_loss / len(loader)
        rmse_loss = rmse_loss / len(loader)
        ssim_loss = ssim_loss / len(loader)
        huberssim_loss = huberssim_loss / len(loader)
        temporal_loss = temporal_loss / len(loader)
        psnr_loss = psnr_loss / len(loader)
        total_loss = total_loss / len(loader)

        if use_wandb:
            wandb.log({f"Checked_{mode}_mae_loss": mae_loss})
            wandb.log({f"Checked_{mode}_mse_loss": mse_loss})
            wandb.log({f"Checked_{mode}_huber_loss": huber_loss})
            wandb.log({f"Checked_{mode}_rmse_loss": rmse_loss})
            wandb.log({f"Checked_{mode}_ssim_loss": ssim_loss})
            wandb.log({f"Checked_{mode}_huberssim_loss": huberssim_loss})
            wandb.log({f"Checked_{mode}_temporal_loss": temporal_loss})
            wandb.log({f"Checked_{mode}_psnr_loss": psnr_loss})
            wandb.log({f"Checked_{mode}_total_loss": total_loss})
        
        return mae_loss, mse_loss, huber_loss, ssim_loss, huberssim_loss, temporal_loss, rmse_loss, psnr_loss, total_loss
        
    @torch.no_grad()
    def overall_loss(self, loader, mode, use_wandb):
        mae_loss = 0.0
        mse_loss = 0.0
        huber_loss = 0.0
        rmse_loss = 0.0
        ssim_loss = 0.0
        huberssim_loss = 0.0
        temporal_loss = 0.0
        psnr_loss = 0.0
        total_loss = 0.0
        for inputs, targets in loader:
            #inputs = inputs.to(device)
            #targets = targets.to(device)
            outputs = []
            for t in range(0, self.config["pred_frames"], self.config["pred_n_frames_per_step"]):
                if self.config["pred_frames"]-t<self.config["pred_n_frames_per_step"]:
                    frames_this_step = self.config["pred_frames"]-t
                else:
                    frames_this_step = self.config["pred_n_frames_per_step"]
                outputs_t = self.forward(inputs)
                #print(f"{t}:{t+frames_this_step}")
                #get only the first predicted frame
                outputs_t = outputs_t[:, :frames_this_step, :, :, :, :]
                
                outputs.append(outputs_t)
    
                inputs = torch.cat([inputs[:, self.config["pred_n_frames_per_step"]:, :, :, :, :], outputs_t], dim=1)
                
                #concat time and add to overall lst
            outputs = torch.concat(outputs, dim=1)
            
            #calculate losses
            mae_loss += self.mae_criterion(outputs, targets).item()
            mse_loss += self.mse_criterion(outputs, targets).item()
            rmse_loss += torch.sqrt(self.mse_criterion(outputs, targets)).item()
            huberssim_loss_, huber_loss_, ssim_loss_, temporal_loss_ = self.huberssim3d_criterion(outputs, targets)
            huberssim_loss += huberssim_loss_.item()
            ssim_loss += ssim_loss_.item()
            temporal_loss += temporal_loss_.item()
            huber_loss += huber_loss_.item()
            psnr_loss += model.psnr_criterion(outputs, targets).item()
            total_loss += mse_loss + 0.5 * huber_loss
    
        mae_loss = mae_loss / len(loader)
        mse_loss = mse_loss / len(loader)
        rmse_loss = rmse_loss / len(loader)
        huberssim_loss = huberssim_loss / len(loader)
        huber_loss = huber_loss / len(loader)
        ssim_loss = ssim_loss / len(loader)
        temporal_loss = temporal_loss / len(loader)
        psnr_loss = psnr_loss / len(loader)
        total_loss = total_loss / len(loader)

        if use_wandb:
            wandb.log({f"Overall_{mode}_mae_loss": mae_loss})
            wandb.log({f"Overall_{mode}_mse_loss": mse_loss})
            wandb.log({f"Overall_{mode}_huber_loss": huber_loss})
            wandb.log({f"Overall_{mode}_rmse_loss": rmse_loss})
            wandb.log({f"Overall_{mode}_ssim_loss": ssim_loss})
            wandb.log({f"Overall_{mode}_huberssim_loss": huberssim_loss})
            wandb.log({f"Overall_{mode}_temporal_loss": temporal_loss})
            wandb.log({f"Overall_{mode}_psnr_loss": psnr_loss})
            wandb.log({f"Overall_{mode}_total_loss": total_loss})
    
        return outputs, mae_loss, mse_loss, huber_loss, rmse_loss, ssim_loss, huberssim_loss, temporal_loss, psnr_loss, total_loss

    @torch.no_grad()
    def predict_one_ct(self, path, input_frames, pred_frames, pred_n_frames_per_step):
        inputs = torch.from_numpy(np.load(path)).unsqueeze(1).unsqueeze(0)
        targets = inputs.clone()[:, input_frames:input_frames+pred_frames, :, :, :, :]
        inputs = inputs[:, :input_frames, :, :, :, :]
        
        outputs = []
        for t in range(0, pred_frames, pred_n_frames_per_step):
            if pred_frames-t<pred_n_frames_per_step:
                frames_this_step = pred_frames-t
            else:
                frames_this_step = pred_n_frames_per_step
            outputs_t = self.forward(inputs)
            #print(f"{t}:{t+frames_this_step}")
            #get only the first predicted frame
            outputs_t = outputs_t[:, :frames_this_step, :, :, :, :]
            
            outputs.append(outputs_t)

            inputs = torch.cat([inputs[:, pred_n_frames_per_step:, :, :, :, :], outputs_t], dim=1)
            
            #concat time and add to overall lst
        outputs = torch.concat(outputs, dim=1)

        #concat and squeeze batch_dim=1 and channel_dim=1
        targets = torch.concat([inputs, targets], dim=1).squeeze(2).squeeze(0)
        outputs = torch.concat([inputs, outputs], dim=1).squeeze(2).squeeze(0)

        return outputs, targets