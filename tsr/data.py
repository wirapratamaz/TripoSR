import os
import torch
import numpy as np
from PIL import Image
import trimesh
import glob
from torch.utils.data import Dataset, DataLoader

class TripoSRDataset(Dataset):
    def __init__(self, data_path, resolution=128, transform=None):
        """
        Dataset for TripoSR fine-tuning
        
        Args:
            data_path (str): Path to dataset directory
            resolution (int): Resolution for input images and target meshes
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.data_path = data_path
        self.resolution = resolution
        self.transform = transform
        
        # Get all object IDs (folder names) in the dataset
        self.object_ids = []
        for obj_dir in os.listdir(data_path):
            # Skip hidden files/directories
            if obj_dir.startswith('.'):
                continue
                
            obj_path = os.path.join(data_path, obj_dir)
            if os.path.isdir(obj_path):
                # Check if directory contains required files
                if (os.path.exists(os.path.join(obj_path, "image.png")) and 
                    (os.path.exists(os.path.join(obj_path, "model.obj")) or
                     os.path.exists(os.path.join(obj_path, "model.ply")) or
                     os.path.exists(os.path.join(obj_path, "model.off")))):
                    self.object_ids.append(obj_dir)
        
        print(f"Found {len(self.object_ids)} valid objects in {data_path}")
    
    def __len__(self):
        return len(self.object_ids)
    
    def __getitem__(self, idx):
        obj_id = self.object_ids[idx]
        obj_path = os.path.join(self.data_path, obj_id)
        
        # Load image
        image_path = os.path.join(obj_path, "image.png")
        image = Image.open(image_path).convert("RGB")
        
        # Resize image to the required resolution
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to CxHxW format
        
        # Load 3D mesh (could be in different formats)
        mesh_path = None
        for ext in [".obj", ".ply", ".off"]:
            potential_path = os.path.join(obj_path, f"model{ext}")
            if os.path.exists(potential_path):
                mesh_path = potential_path
                break
        
        mesh = trimesh.load(mesh_path)
        
        # Check for mask
        mask_path = os.path.join(obj_path, "mask.png")
        mask = None
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        # Convert mesh to tensor format
        vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
        faces = torch.from_numpy(mesh.faces.astype(np.int64))
        
        return {
            "image": image,
            "vertices": vertices,
            "faces": faces,
            "mask": mask,
            "obj_id": obj_id
        }


def collate_fn(batch):
    """
    Custom collate function for variable sized meshes
    """
    images = torch.stack([item["image"] for item in batch])
    
    # For the mesh data, we can't easily batch them as they have different sizes
    vertices = [item["vertices"] for item in batch]
    faces = [item["faces"] for item in batch]
    
    # Masks might be None for some samples
    masks = []
    for item in batch:
        if item["mask"] is not None:
            masks.append(item["mask"])
        else:
            # Create an empty mask if none is provided
            masks.append(torch.ones((1, batch[0]["image"].shape[1], batch[0]["image"].shape[2])))
    masks = torch.stack(masks) if masks else None
    
    obj_ids = [item["obj_id"] for item in batch]
    
    return {
        "images": images,
        "vertices": vertices,
        "faces": faces, 
        "masks": masks,
        "obj_ids": obj_ids
    }


def get_dataloaders(config):
    """
    Create data loaders for training and validation
    
    Args:
        config: Configuration object with data paths and parameters
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TripoSRDataset(
        data_path=config.data.train_path,
        resolution=config.data.resolution
    )
    
    val_dataset = TripoSRDataset(
        data_path=config.data.val_path,
        resolution=config.data.resolution
    )
    
    # Get number of workers from config if available, otherwise use default value
    num_workers = getattr(config.data, "num_workers", 4) if hasattr(config, "data") else 4
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader 