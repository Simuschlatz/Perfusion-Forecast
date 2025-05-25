import os
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

def set_seed(seed=100):
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    np.random.seed(seed)
    random.seed(seed)

class Dataset2D(Dataset):
    def __init__(
        self, data_paths, context_window=4, prediction_window=1, transform=None
    ):
        self.data_paths = data_paths
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.transform = transform
        self.depth = None
        # For every path to a volume sequence in .npy
        self.data_attributes = []
        test = np.load(data_paths[0])
        for data_path in self.data_paths:
            for h in range(test.shape[1]):
                for t in range(len(test)-self.context_window - self.prediction_window+1):
                    # file_path, t, h
                    self.data_attributes.append([data_path, t, h])

    def __len__(self):
        return len(self.data_attributes)

    def __getitem__(self, idx):
        t = self.data_attributes[idx][1]
        h = self.data_attributes[idx][2]
        volume_seq = torch.from_numpy(np.load(self.data_attributes[idx][0]))
        return (
            volume_seq[t:t+self.context_window, h].unsqueeze(1),
            volume_seq[t+self.context_window:t+self.context_window+self.prediction_window, h].unsqueeze(1)
        )

class Dataset3D(Dataset):
    def __init__(self, data_paths, context_window=4, prediction_window=1, transform=None):
        self.data_paths = data_paths
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.transform = transform

        self.data_attributes = []
        test = np.load(self.data_paths[0])
        for data_path in self.data_paths:
            for t in range(test.shape[0]-self.context_window-self.prediction_window+1):
                # file_path, t
                self.data_attributes.append([data_path, t])
        
    def __len__(self):
        return len(self.data_attributes)
    
    def __getitem__(self, idx):
        t = self.data_attributes[idx][1]
        volume_seq = torch.from_numpy(np.load(self.data_attributes[idx][0]))
        return (
            volume_seq[t:t+self.context_window].unsqueeze(1), 
            volume_seq[t+self.context_window:t+self.context_window+self.prediction_window].unsqueeze(1)
        )

def seed_worker(worker_id):
    """Ensures worker processes get the same seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class VolumeDataModule2D(pl.LightningDataModule):
    def __init__(self, root, batch_size=4, sequence_length=4, prediction_length=1, num_workers=0, drop_last=False, pin_memory=False, train_split=0.8, val_split=0.1, test_split=0.1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None

    def setup(self, stage=None):
        data_paths = [os.path.join(self.root, path) for path in os.listdir(self.root)]
        total_size = len(data_paths)
    
        # Normalize splits if they don’t sum to 1
        split_sum = self.train_split + self.val_split + self.test_split
        if split_sum != 1.0:
            self.train_split /= split_sum
            self.val_split /= split_sum
            self.test_split /= split_sum
            print(f"Normalized splits to: train={self.train_split:.2f}, val={self.val_split:.2f}, test={self.test_split:.2f}")
    
        # Compute dataset sizes
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size  # Ensure all data is used
    
        # Error handling: Ensure valid split sizes
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(f"Invalid dataset splits: train={train_size}, val={val_size}, test={test_size}. Check your split values.")
    
        # Perform random split
        self.train_paths, self.val_paths, self.test_paths = random_split(data_paths, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

        self.train_dataset = Dataset2D(self.train_paths, self.sequence_length, self.prediction_length)
        self.val_dataset = Dataset2D(self.val_paths, self.sequence_length, self.prediction_length)
        self.test_dataset = Dataset2D(self.test_paths, self.sequence_length, self.prediction_length)
    
        self.val_dataset_3d = Dataset3D(self.val_paths, self.sequence_length, self.prediction_length)
        self.test_dataset_3d = Dataset3D(self.test_paths, self.sequence_length, self.prediction_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker)

    def val_dataloader(self):
        return [
            DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker),
            DataLoader(self.val_dataset_3d, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker),
        ]

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker)

    def val_dataloader_3d(self):
        return DataLoader(self.val_dataset_3d, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker)
    
    def test_dataloader_3d(self):
        return DataLoader(self.test_dataset_3d, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker)

    def teardown(self, stage=None):
        if stage == "fit" or stage is None:
            pass
            #print("Cleaning up after training...")

        if stage == "test" or stage is None:
            pass
            #print("Cleaning up after testing...")

        if stage == "validate" or stage is None:
            pass
            #print("Cleaning up after validation...")

        # Free memory by deleting large datasets
        del self.train_dataset
        del self.val_dataset
        del self.test_dataset
        del self.val_dataset_3d
        del self.test_dataset_3d

class VolumeDataModule3D(pl.LightningDataModule):
    def __init__(self, root, batch_size=4, sequence_length=4, prediction_length=1, num_workers=0, drop_last=False, pin_memory=False, train_split=0.8, val_split=0.1, test_split=0.1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        data_paths = [os.path.join(self.root, path) for path in os.listdir(self.root)]
        total_size = len(data_paths)
    
        # Normalize splits if they don’t sum to 1
        split_sum = self.train_split + self.val_split + self.test_split
        if split_sum != 1.0:
            self.train_split /= split_sum
            self.val_split /= split_sum
            self.test_split /= split_sum
            print(f"Normalized splits to: train={self.train_split:.2f}, val={self.val_split:.2f}, test={self.test_split:.2f}")
    
        # Compute dataset sizes
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size  # Ensure all data is used
    
        # Error handling: Ensure valid split sizes
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(f"Invalid dataset splits: train={train_size}, val={val_size}, test={test_size}. Check your split values.")
    
        # Perform random split
        self.train_paths, self.val_paths, self.test_paths = random_split(data_paths, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

        
        
        self.train_dataset = Dataset3D(self.train_paths, self.sequence_length, self.prediction_length)
        self.val_dataset = Dataset3D(self.val_paths, self.sequence_length, self.prediction_length)
        self.test_dataset = Dataset3D(self.test_paths, self.sequence_length, self.prediction_length)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory, worker_init_fn=seed_worker)

    def teardown(self, stage=None):
        if stage == "fit" or stage is None:
            pass
            #print("Cleaning up after training...")

        if stage == "test" or stage is None:
            pass
            #print("Cleaning up after testing...")

        if stage == "validate" or stage is None:
            pass
            #print("Cleaning up after validation...")

        # Free memory by deleting large datasets
        del self.train_dataset
        del self.val_dataset
        del self.test_dataset