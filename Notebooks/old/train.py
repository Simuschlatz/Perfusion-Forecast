import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Config, get_data_loaders
from nnets import PerfusionPredictor
from tqdm import tqdm
import wandb
from icecream import ic

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss /\
        len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def train_model(config=Config):
    # Initialize wandb
    wandb.init(project="perfusion-prediction")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
    )
    
    # Initialize model
    model = PerfusionPredictor(sequence_length=config.sequence_length)
    model = model.to(config.device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        val_loss = validate(model, val_loader, criterion, config.device)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_model()