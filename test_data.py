import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
import trimesh

from tsr.data import TripoSRDataset, get_dataloaders


def test_dataset():
    """Test the dataset loading and visualization"""
    parser = argparse.ArgumentParser(description="Test TripoSR dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="dataset_test", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create train and validation datasets
    print("Loading datasets...")
    train_dataset = TripoSRDataset(
        data_path=config.data.train_path,
        resolution=config.data.resolution
    )
    
    val_dataset = TripoSRDataset(
        data_path=config.data.val_path,
        resolution=config.data.resolution
    )
    
    print(f"Found {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Check dataset structure
    print("\nChecking dataset structure...")
    check_dataset_structure(config.data.train_path, "training")
    check_dataset_structure(config.data.val_path, "validation")
    
    # Visualize random samples
    print("\nVisualizing random samples...")
    
    # Training samples
    visualize_samples(train_dataset, 
                      min(args.num_samples, len(train_dataset)), 
                      os.path.join(args.output_dir, "train"),
                      "Training")
    
    # Validation samples
    visualize_samples(val_dataset, 
                      min(args.num_samples, len(val_dataset)), 
                      os.path.join(args.output_dir, "val"),
                      "Validation")
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    
    # Load a batch from each
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    print(f"\nTraining batch contains:")
    print_batch_info(train_batch)
    
    print(f"\nValidation batch contains:")
    print_batch_info(val_batch)
    
    print("\nDataset test completed successfully!")


def check_dataset_structure(data_path, split_name):
    """Check if the dataset is properly structured"""
    if not os.path.exists(data_path):
        print(f"WARNING: {split_name} dataset path {data_path} does not exist!")
        return
    
    objects = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if not objects:
        print(f"WARNING: No object directories found in {split_name} dataset!")
        return
    
    print(f"Found {len(objects)} object directories in {split_name} dataset")
    
    # Check a few samples
    samples_to_check = min(5, len(objects))
    issues = []
    
    for obj_id in objects[:samples_to_check]:
        obj_path = os.path.join(data_path, obj_id)
        
        # Check for image
        if not os.path.exists(os.path.join(obj_path, "image.png")):
            issues.append(f"Missing image.png in {obj_id}")
        
        # Check for mesh
        mesh_found = False
        for ext in [".obj", ".ply", ".off"]:
            if os.path.exists(os.path.join(obj_path, f"model{ext}")):
                mesh_found = True
                break
        
        if not mesh_found:
            issues.append(f"Missing model mesh file in {obj_id}")
    
    if issues:
        print(f"Found issues in {split_name} dataset:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print(f"No issues found in the checked {split_name} samples")


def visualize_samples(dataset, num_samples, output_dir, title):
    """Visualize random samples from the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Create figure for visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Plot image
        plt.subplot(1, 3, 1)
        if isinstance(sample["image"], torch.Tensor):
            image = sample["image"].permute(1, 2, 0).numpy()
        else:
            image = sample["image"]
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis("off")
        
        # Plot mask if available
        plt.subplot(1, 3, 2)
        if sample["mask"] is not None:
            if isinstance(sample["mask"], torch.Tensor):
                mask = sample["mask"].squeeze().numpy()
            else:
                mask = sample["mask"]
            plt.imshow(mask, cmap="gray")
            plt.title("Mask")
        else:
            plt.title("No Mask Available")
        plt.axis("off")
        
        # Plot mesh wireframe
        plt.subplot(1, 3, 3)
        vertices = sample["vertices"].numpy() if isinstance(sample["vertices"], torch.Tensor) else sample["vertices"]
        faces = sample["faces"].numpy() if isinstance(sample["faces"], torch.Tensor) else sample["faces"]
        
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        
        # Plot wireframe
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            alpha=0.5,
            color='blue'
        )
        
        ax.set_title("3D Mesh")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i}_{sample['obj_id']}.png"))
        plt.close()
        
        # Also save the mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(os.path.join(output_dir, f"sample_{i}_{sample['obj_id']}.obj"))
    
    print(f"Saved {num_samples} {title.lower()} sample visualizations to {output_dir}")


def print_batch_info(batch):
    """Print information about a batch from the dataloader"""
    print(f"- Images: {batch['images'].shape}")
    print(f"- Vertices: {len(batch['vertices'])} meshes")
    print(f"- Faces: {len(batch['faces'])} meshes")
    if batch['masks'] is not None:
        print(f"- Masks: {batch['masks'].shape}")
    print(f"- Object IDs: {batch['obj_ids']}")
    
    # Print some statistics about the meshes
    vertices_counts = [v.shape[0] for v in batch['vertices']]
    faces_counts = [f.shape[0] for f in batch['faces']]
    
    print(f"- Vertices per mesh: min={min(vertices_counts)}, max={max(vertices_counts)}, avg={sum(vertices_counts)/len(vertices_counts):.1f}")
    print(f"- Faces per mesh: min={min(faces_counts)}, max={max(faces_counts)}, avg={sum(faces_counts)/len(faces_counts):.1f}")


if __name__ == "__main__":
    test_dataset() 