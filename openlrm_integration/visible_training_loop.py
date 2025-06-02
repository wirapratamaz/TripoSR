import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import clear_output, display
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

class VisibleTrainingLoop:
    """
    A training loop implementation for TripoSR with OpenLRM that provides
    real-time visibility into the training process with progress bars and metrics.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader=None,
        device='cuda',
        epochs=10,
        save_dir='./checkpoints',
        export_ckpt=True,
        log_interval=10,
        use_amp=True
    ):
        """
        Initialize the training loop.
        
        Args:
            model: The PyTorch model to train
            optimizer: The optimizer to use
            criterion: The loss function
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            device: Device to train on ('cuda' or 'cpu')
            epochs: Number of epochs to train for
            save_dir: Directory to save checkpoints
            export_ckpt: Whether to export .ckpt files (compatible with OpenLRM)
            log_interval: How often to log metrics (in batches)
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.export_ckpt = export_ckpt
        self.log_interval = log_interval
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Create directory for saving checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {'loss': []}
        self.val_metrics = {'loss': []}
        self.batch_times = []
        self.epoch_times = []
        
        # Initialize scaler for mixed precision training
        self.scaler = GradScaler() if self.use_amp else None
        
        # For plotting
        self.fig = None
        self.axs = None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        samples_seen = 0
        epoch_start_time = time.time()
        
        # Create progress bar for batches
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=False,
            unit="batch"
        )
        
        for i, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move data to device
            if isinstance(batch, list) or isinstance(batch, tuple):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                # Handle case where batch is a dictionary
                inputs = batch
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(self.device)
                targets = None
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    if targets is not None:
                        loss = self.criterion(outputs, targets)
                    else:
                        # Handle case where model returns loss directly
                        loss = outputs if isinstance(outputs, torch.Tensor) else outputs['loss']
                
                # Backward and optimize with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs = self.model(inputs)
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    # Handle case where model returns loss directly
                    loss = outputs if isinstance(outputs, torch.Tensor) else outputs['loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            running_loss += batch_loss
            
            # Batch size might vary in the last batch
            batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else inputs[list(inputs.keys())[0]].size(0)
            samples_seen += batch_size
            
            # Calculate batch processing time
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)
            
            # Calculate samples per second
            samples_per_sec = batch_size / batch_time
            
            # Update progress bar
            pbar.set_postfix({
                'loss': batch_loss,
                'avg_loss': running_loss / (i + 1),
                'samples/sec': f"{samples_per_sec:.2f}"
            })
            
            # Log periodically
            if i % self.log_interval == 0:
                self.train_metrics['loss'].append(batch_loss)
                # Update plots if in a notebook environment
                self._update_plots()
        
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        
        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss, samples_seen / epoch_time
    
    def validate(self, epoch):
        """Validate the model"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Validation {epoch+1}/{self.epochs}",
            leave=False,
            unit="batch"
        )
        
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                # Move data to device
                if isinstance(batch, list) or isinstance(batch, tuple):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    # Handle case where batch is a dictionary
                    inputs = batch
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            inputs[k] = inputs[k].to(self.device)
                    targets = None
                
                # Forward pass
                outputs = self.model(inputs)
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    # Handle case where model returns loss directly
                    loss = outputs if isinstance(outputs, torch.Tensor) else outputs['loss']
                
                # Update metrics
                batch_loss = loss.item()
                val_loss += batch_loss
                
                # Update progress bar
                pbar.set_postfix({'val_loss': batch_loss, 'avg_val_loss': val_loss / (i + 1)})
        
        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        self.val_metrics['loss'].append(avg_val_loss)
        
        val_time = time.time() - val_start_time
        
        return avg_val_loss, val_time
    
    def train(self):
        """Run the full training loop"""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Using device: {self.device}")
        print(f"Mixed precision: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Main training loop with progress bar for epochs
        for epoch in tqdm(range(self.epochs), desc="Training Progress", unit="epoch"):
            # Train for one epoch
            train_loss, train_throughput = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_time = self.validate(epoch)
            
            # Log epoch results
            print(f"\nEpoch {epoch+1}/{self.epochs} completed:")
            print(f"  Train Loss: {train_loss:.6f}")
            if self.val_loader:
                print(f"  Validation Loss: {val_loss:.6f}")
            print(f"  Training Speed: {train_throughput:.2f} samples/sec")
            print(f"  GPU Memory: {self._get_gpu_memory_usage():.2f} MB")
            
            # Save checkpoint
            self._save_checkpoint(epoch, train_loss, val_loss)
            
            # Update plots
            self._update_plots(force=True)
        
        print("\nTraining completed!")
        return self.train_losses, self.val_losses
    
    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # Save regular PyTorch checkpoint
        torch.save(checkpoint, f"{self.save_dir}/checkpoint_epoch_{epoch+1}.pt")
        
        # Save in OpenLRM .ckpt format if requested
        if self.export_ckpt:
            # Assuming OpenLRM expects just the model weights in .ckpt format
            torch.save(self.model.state_dict(), f"{self.save_dir}/model_epoch_{epoch+1}.ckpt")
        
        print(f"Checkpoint saved at epoch {epoch+1}")
    
    def _update_plots(self, force=False):
        """Update training plots"""
        # Only update every log_interval or when forced
        if not force and len(self.train_metrics['loss']) % self.log_interval != 0:
            return
        
        # Create or clear the figure
        if self.fig is None or force:
            clear_output(wait=True)
            self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training loss
        ax = self.axs[0, 0]
        ax.clear()
        ax.plot(self.train_metrics['loss'], label='Training Loss')
        ax.set_title('Training Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Plot epoch losses
        ax = self.axs[0, 1]
        ax.clear()
        epochs = list(range(1, len(self.train_losses) + 1))
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        if self.val_losses:
            ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax.set_title('Loss per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Plot throughput
        ax = self.axs[1, 0]
        ax.clear()
        if self.batch_times:
            batch_throughputs = [1.0 / t for t in self.batch_times[-100:]]  # Last 100 batches
            ax.plot(batch_throughputs)
            ax.set_title('Training Throughput (batches/sec)')
            ax.set_xlabel('Last 100 Batches')
            ax.set_ylabel('Batches per Second')
            ax.grid(True)
        
        # Plot GPU memory usage
        ax = self.axs[1, 1]
        ax.clear()
        memory_usage = self._get_gpu_memory_usage()
        ax.bar(['Current'], [memory_usage])
        ax.set_title('GPU Memory Usage')
        ax.set_ylabel('Memory (MB)')
        ax.set_ylim(0, torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        
        plt.tight_layout()
        display(self.fig)
    
    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0
        
        # Get current GPU memory usage
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)


# Example usage:
"""
# Create model, optimizer, criterion, dataloaders
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create the visible training loop
trainer = VisibleTrainingLoop(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    save_dir='./checkpoints',
    export_ckpt=True
)

# Start training
train_losses, val_losses = trainer.train()
"""
