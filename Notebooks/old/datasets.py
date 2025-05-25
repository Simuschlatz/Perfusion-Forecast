import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from preprocessing import get_volume
from icecream import ic

class Config:
    dataset_path = os.path.expanduser('~/Desktop/UniToBrain')
    device = torch.device('mps') or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training parameters
    batch_size = 4
    sequence_length = 4
    learning_rate = 1e-4
    num_epochs = 100
    
    # Data parameters
    train_split = 0.8
    val_split = 0.1
    # test_split will be the remainder

class Dataset3D(Dataset):
    def __init__(self, source_path, context_window=4, transform=None):
        self.data_paths = os.listdir(source_path)
        self.context_window = context_window
        self.transform = transform
        self.samples = []
        for data_path in self.data_paths:
            volume_seq = np.load(os.path.join(data_path))
            volume_seq = torch.from_numpy(volume_seq)
            for i in range(len(volume_seq) - self.context_window):
                # Input volume sequence (context_window x 16 x 256 x 256), target volume (16 x 256 x 256)
                self.samples.append((volume_seq[i:i+self.context_window], volume_seq[i+self.context_window]))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def get_data_loaders(batch_size=4, sequence_length=4):
    # Load all folder paths
    folder_paths = []
    for folder in sorted(os.listdir(Config.dataset_path)):
        folder_path = os.path.join(Config.dataset_path, folder)
        if len(folder) == 7:  # MOL-XYZ format
            folder_paths.append(folder_path)
            
    # Split into train/val/test
    n_train = int(len(folder_paths) * Config.train_split)
    n_val = int(len(folder_paths) * Config.val_split)
    
    train_paths = folder_paths[:n_train]
    val_paths = folder_paths[n_train:n_train+n_val]
    test_paths = folder_paths[n_train+n_val:]
    
    # Create datasets
    train_dataset = Dataset3D(train_paths, sequence_length)
    val_dataset = Dataset3D(val_paths, sequence_length)
    test_dataset = Dataset3D(test_paths, sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
        