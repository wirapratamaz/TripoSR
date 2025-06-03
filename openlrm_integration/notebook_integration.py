"""
Integration example for the visible training loop with TripoSR and OpenLRM.
This script demonstrates how to incorporate the visible_training_loop.py into 
the existing TripoSR OpenLRM training notebook.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Import the visible training loop
from visible_training_loop import VisibleTrainingLoop

def integrate_visible_training(config_path, dataset_path, model_class, dataset_class):
    """
    Integrate the visible training loop with TripoSR and OpenLRM.
    
    Args:
        config_path: Path to the configuration file
        dataset_path: Path to the dataset directory
        model_class: The model class to use
        dataset_class: The dataset class to use
    
    Returns:
        Trained model and training metrics
    """
    print("Setting up visible training loop for TripoSR with OpenLRM...")
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    print(f"Loaded configuration from {config_path}")
    
    # Update dataset path if provided
    if dataset_path:
        cfg.data.dataset_path = dataset_path
        print(f"Using dataset from: {dataset_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = model_class(cfg).to(device)
    print("Model created")
    
    # Create optimizer
    if cfg.optimizer.name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
    
    print(f"Using optimizer: {cfg.optimizer.name} with lr={cfg.optimizer.lr}")
    
    # Create dataset and dataloader
    train_dataset = dataset_class(cfg, split='train')
    val_dataset = dataset_class(cfg, split='val') if cfg.data.get('val_split', True) else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
    
    print(f"Training dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create loss function
    # This is a placeholder - you'll need to use the actual loss function from your model
    criterion = lambda x, y: x  # Assuming model returns loss directly
    
    # Setup checkpoint directory
    os.makedirs(cfg.saver.checkpoint_dir, exist_ok=True)
    
    # Create visible training loop
    trainer = VisibleTrainingLoop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg.train.epochs,
        save_dir=cfg.saver.checkpoint_dir,
        export_ckpt=True,
        log_interval=cfg.logger.log_interval if hasattr(cfg.logger, 'log_interval') else 10,
        use_amp=cfg.train.get('use_amp', True)
    )
    
    # Start training
    train_losses, val_losses = trainer.train()
    
    # Export final model checkpoint in OpenLRM format
    final_ckpt_path = os.path.join(cfg.saver.checkpoint_dir, "final_model.ckpt")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Final model saved to {final_ckpt_path}")
    
    return model, (train_losses, val_losses)


# Example usage in notebook:
"""
# Import the integration module
import sys
sys.path.append('./TripoSR')
from notebook_integration import integrate_visible_training

# Import the necessary model and dataset classes from OpenLRM and TripoSR
from openlrm.models import TripoSRModel  # Replace with the actual import path
from openlrm.datasets import TripoSRDataset  # Replace with the actual import path

# Path to configuration and dataset
config_path = 'TripoSR/openlrm_integration/configs/colab_config.yaml'
dataset_path = './local_dataset'  # Point to your local .glb format dataset

# Run the visible training loop
model, (train_losses, val_losses) = integrate_visible_training(
    config_path=config_path,
    dataset_path=dataset_path,
    model_class=TripoSRModel,
    dataset_class=TripoSRDataset
)

# The training progress will be displayed with progress bars and plots
# Once training is complete, the model will be saved in the checkpoint directory
# specified in the configuration file
"""
