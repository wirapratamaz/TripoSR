import os
import argparse
import logging
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf

from tsr.system import TSR
from tsr.data import get_dataloaders
from tsr.utils import load_config
from tsr.evaluation import chamfer_distance


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def mesh_loss(pred_vertices, pred_faces, target_vertices, target_faces):
    """
    Compute loss between predicted mesh and target mesh
    
    Args:
        pred_vertices (torch.Tensor): Predicted vertices
        pred_faces (torch.Tensor): Predicted faces
        target_vertices (torch.Tensor): Target vertices
        target_faces (torch.Tensor): Target faces
    
    Returns:
        torch.Tensor: Loss value
    
    Note:
        This implementation uses Chamfer distance as the primary loss metric.
        Potential alternatives or additions to consider for more refined training:
        - Edge length consistency loss
        - Normal consistency loss
        - Laplacian smoothing loss
        - IoU-based loss
        
        The mesh extraction process in each training step is computationally expensive.
        For optimization, one could potentially define a loss directly on the model's
        internal representation (triplane or density field) if the architecture allows.
    """
    # Compute Chamfer distance between predicted and target vertices
    return chamfer_distance(pred_vertices.unsqueeze(0), target_vertices.unsqueeze(0))


def train_epoch(model, train_loader, optimizer, device, epoch, config):
    """
    Train for one epoch
    
    Args:
        model (TSR): TripoSR model
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        device (str): Device to use
        epoch (int): Current epoch
        config (OmegaConf): Configuration
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.epochs}")
    
    for i, batch in enumerate(pbar):
        # Move data to device
        images = batch["images"].to(device)
        vertices_list = [v.to(device) for v in batch["vertices"]]
        faces_list = [f.to(device) for f in batch["faces"]]
        
        # Forward pass
        scene_codes = model(images, device=device)
        
        # Extract meshes from scene codes
        pred_meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
        
        # Compute loss
        batch_loss = 0
        for j, (pred_mesh, target_vertices, target_faces) in enumerate(
            zip(pred_meshes, vertices_list, faces_list)
        ):
            loss = mesh_loss(
                torch.tensor(pred_mesh.vertices, device=device),
                torch.tensor(pred_mesh.faces, device=device),
                target_vertices,
                target_faces
            )
            batch_loss += loss
        
        batch_loss = batch_loss / len(pred_meshes)
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        
        # Log progress
        if (i + 1) % config.training.log_interval == 0:
            pbar.set_postfix({"loss": batch_loss.item()})
            logger.info(
                f"Epoch {epoch}/{config.training.epochs} | "
                f"Iteration {i+1}/{len(train_loader)} | "
                f"Loss: {batch_loss.item():.6f}"
            )
    
    # Compute average loss
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch}/{config.training.epochs} | Average loss: {avg_loss:.6f}")
    
    return avg_loss


def validate(model, val_loader, device, epoch, config):
    """
    Validate the model
    
    Args:
        model (TSR): TripoSR model
        val_loader (DataLoader): Validation data loader
        device (str): Device to use
        epoch (int): Current epoch
        config (OmegaConf): Configuration
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            # Move data to device
            images = batch["images"].to(device)
            vertices_list = [v.to(device) for v in batch["vertices"]]
            faces_list = [f.to(device) for f in batch["faces"]]
            
            # Forward pass
            scene_codes = model(images, device=device)
            
            # Extract meshes from scene codes
            pred_meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
            
            # Compute loss
            batch_loss = 0
            for j, (pred_mesh, target_vertices, target_faces) in enumerate(
                zip(pred_meshes, vertices_list, faces_list)
            ):
                loss = mesh_loss(
                    torch.tensor(pred_mesh.vertices, device=device),
                    torch.tensor(pred_mesh.faces, device=device),
                    target_vertices,
                    target_faces
                )
                batch_loss += loss
            
            batch_loss = batch_loss / len(pred_meshes)
            total_loss += batch_loss.item()
    
    # Compute average loss
    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Epoch {epoch}/{config.training.epochs} | Average loss: {avg_loss:.6f}")
    
    return avg_loss


def visualize_results(model, val_loader, device, epoch, output_dir):
    """
    Visualize some validation results
    
    Args:
        model (TSR): TripoSR model
        val_loader (DataLoader): Validation data loader
        device (str): Device to use
        epoch (int): Current epoch
        output_dir (str): Output directory
    """
    model.eval()
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, f"vis_epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get a batch from validation loader
    batch = next(iter(val_loader))
    images = batch["images"].to(device)
    obj_ids = batch["obj_ids"]
    
    with torch.no_grad():
        # Generate scene codes
        scene_codes = model(images, device=device)
        
        # Render images
        render_images = model.render(scene_codes, n_views=4, return_type="np")
        
        # Extract meshes
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True)
    
    # Save images, renders and meshes
    for i, (image, renders, mesh, obj_id) in enumerate(
        zip(images.cpu().numpy(), render_images, meshes, obj_ids)
    ):
        # Save input image
        plt.figure(figsize=(5, 5))
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(vis_dir, f"{obj_id}_input.png"))
        plt.close()
        
        # Save rendered views
        fig, axes = plt.subplots(1, len(renders), figsize=(15, 5))
        for j, render in enumerate(renders):
            axes[j].imshow(render)
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{obj_id}_renders.png"))
        plt.close()
        
        # Save mesh
        mesh.export(os.path.join(vis_dir, f"{obj_id}_mesh.obj"))


def train():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train TripoSR model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        logger.warning("CUDA not available, using CPU instead")
        device = "cpu"
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Initialize model
    logger.info("Initializing model...")
    if args.pretrained:
        # Load pretrained model for transfer learning
        logger.info("Loading pretrained model...")
        model = TSR.from_pretrained(
            "TrianC0de/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
    else:
        # Initialize model from config
        model = TSR(config)
    
    model.to(device)
    
    # Set up data loaders
    logger.info("Setting up data loaders...")
    train_loader, val_loader = get_dataloaders(config)
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config.training.epochs,
        eta_min=config.training.learning_rate * 0.1
    )
    
    # Training loop
    logger.info("Starting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(1, config.training.epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, config)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Visualize results
        if epoch % 10 == 0:
            visualize_results(model, val_loader, device, epoch, args.output_dir)
        
        # Save checkpoint
        if epoch % config.training.save_interval == 0:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save latest checkpoint
            latest_path = os.path.join(checkpoint_dir, "model_latest.pth")
            torch.save(model.state_dict(), latest_path)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_plot.png"))
    plt.close()


if __name__ == "__main__":
    train() 