"""
OpenLRM trainer implementation for TripoSR.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import safetensors.torch

from openlrm.runners.train.base_trainer import Trainer
from openlrm.utils.logging import get_logger

from ..models.model import build_model
from ..data.dataset import build_dataloader

logger = get_logger(__name__)


class TripoSRTrainer(Trainer):
    """
    Trainer for TripoSR with OpenLRM integration.
    Extends the base Trainer class from OpenLRM.
    """
    
    def __init__(self):
        """Initialize the trainer."""
        super().__init__()
        
        # Build model, optimizer, scheduler, and dataloaders
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._build_dataloader()
        self._build_loss_fn()
    
    def _build_model(self):
        """Build the model."""
        logger.info("Building model...")
        self.model = build_model(self.cfg)
        logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
    
    def _build_optimizer(self):
        """Build the optimizer."""
        logger.info("Building optimizer...")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            betas=(self.cfg.train.beta1, self.cfg.train.beta2),
            weight_decay=self.cfg.train.weight_decay,
            eps=self.cfg.train.eps
        )
        logger.info(f"Optimizer built: {type(self.optimizer).__name__}")
    
    def _build_scheduler(self):
        """Build the learning rate scheduler."""
        logger.info("Building scheduler...")
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.train.epochs,
            eta_min=self.cfg.train.min_lr
        )
        logger.info(f"Scheduler built: {type(self.scheduler).__name__}")
    
    def _build_dataloader(self):
        """Build the dataloaders for training and validation."""
        logger.info("Building dataloaders...")
        self.train_loader = build_dataloader(self.cfg, split='train')
        self.val_loader = build_dataloader(self.cfg, split='val')
        logger.info(f"Dataloaders built: {len(self.train_loader)} training batches, {len(self.val_loader)} validation batches")
    
    def _build_loss_fn(self):
        """Build the loss function."""
        logger.info("Building loss function...")
        self.perceptual_weight = self.cfg.train.loss.perceptual_weight
        
        # Simple MSE loss for demonstration
        self.mse_loss = nn.MSELoss()
        
        logger.info(f"Loss function built with perceptual weight: {self.perceptual_weight}")
    
    def register_hooks(self):
        """Register hooks for the trainer."""
        # No hooks for now
        pass
    
    def train(self):
        """
        Train the model.
        This is the main training loop.
        """
        logger.info("Starting training...")
        
        # Training loop
        for epoch in range(self.current_epoch, self.cfg.train.epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch+1}/{self.cfg.train.epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Calculate loss
                    loss = self._calculate_loss(outputs, batch)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Clip gradients
                    if self.cfg.train.grad_clip > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update training loss
                    train_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                
                # Update global step
                self.global_step += 1
                
                # Log metrics
                self.log_scalar_kwargs(step=self.global_step, loss=loss.item())
                self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0])
                
                # Save checkpoint
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                
                # Evaluate
                if self.global_step % self.cfg.val.eval_global_steps == 0:
                    self.evaluate()
                
                # Check if max steps reached
                if self.global_step >= self.N_max_global_steps:
                    logger.info(f"Reached maximum global steps: {self.global_step}")
                    self.save_checkpoint()
                    return
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            avg_train_loss = train_loss / len(self.train_loader)
            self.log_scalar_kwargs(epoch=epoch, train_loss=avg_train_loss)
            
            # Evaluate at the end of each epoch
            self.evaluate()
        
        # Save final checkpoint
        self.save_checkpoint()
        logger.info("Training completed!")
    
    def _calculate_loss(self, outputs, batch):
        """
        Calculate the loss for a batch.
        
        Args:
            outputs (dict): Model outputs.
            batch (dict): Batch data.
            
        Returns:
            torch.Tensor: Loss value.
        """
        # For demonstration, using a simple MSE loss between tokens
        # In a real implementation, this would be more complex
        tokens = outputs['tokens']
        
        # Calculate reconstruction loss (simplified)
        # In a real implementation, this would involve decoding tokens and comparing with ground truth
        recon_loss = self.mse_loss(tokens, tokens.detach().clone())
        
        # Calculate perceptual loss (simplified)
        # In a real implementation, this would involve comparing features from a pretrained network
        perceptual_loss = torch.tensor(0.0, device=self.device)
        if self.perceptual_weight > 0:
            features = outputs['features']
            perceptual_loss = self.mse_loss(features, features.detach().clone())
        
        # Total loss
        total_loss = recon_loss + self.perceptual_weight * perceptual_loss
        
        return total_loss
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        logger.info("Evaluating model...")
        
        # Set model to evaluation mode
        self.model.eval()
        val_loss = 0.0
        
        # Disable gradient computation
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate loss
                loss = self._calculate_loss(outputs, batch)
                
                # Update validation loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(self.val_loader)
        
        # Log validation metrics
        self.log_scalar_kwargs(step=self.global_step, split='val', loss=avg_val_loss)
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Set model back to training mode
        self.model.train()
    
    def export_model(self, output_path=None):
        """
        Export the trained model to a safetensors file.
        
        Args:
            output_path (str, optional): Path to save the model. If None, use the default path.
        """
        if output_path is None:
            output_path = os.path.join(
                self.cfg.saver.model_root,
                self.cfg.experiment.parent, self.cfg.experiment.child,
                f"model_{self.global_step:06d}.safetensors"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        safetensors.torch.save_model(unwrapped_model, output_path)
        
        logger.info(f"Model exported to {output_path}")
        
        return output_path
