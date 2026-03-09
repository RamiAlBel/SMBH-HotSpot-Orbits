import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict
import wandb
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stop_patience: int = 40,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.early_stop_patience = early_stop_patience
        
        self.best_state = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for batch_x, batch_y in tqdm(
            self.train_loader,
            desc="Training batches",
            leave=False,
        ):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(batch_x).squeeze()
            loss = self.criterion(preds, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(batch_x)
            n_samples += len(batch_x)
        
        return total_loss / n_samples
    
    def validate(self) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(
                self.val_loader,
                desc="Validation batches",
                leave=False,
            ):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                preds = self.model(batch_x).squeeze()
                loss = self.criterion(preds, batch_y)
                
                total_loss += loss.item() * len(batch_x)
                n_samples += len(batch_x)
        
        return total_loss / n_samples
    
    def train(
        self,
        epochs: int,
        use_wandb: bool = False,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Train model with early stopping."""
        epoch_pbar = tqdm(
            range(1, epochs + 1),
            desc="Training epochs",
            leave=True,
        )
        
        for epoch in epoch_pbar:
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'patience': f'{self.patience_counter}/{self.early_stop_patience}'
            })
            
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        epoch_pbar.close()
        
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch
        }
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            **kwargs
        }, path)
