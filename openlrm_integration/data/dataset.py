"""
Dataset implementation for OpenLRM integration with TripoSR.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class TripoSRDataset(Dataset):
    """Dataset for TripoSR with OpenLRM integration."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the dataset.
            split (str): 'train' or 'val' split.
            transform (callable, optional): Optional transform to be applied on samples.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Make sure the directory exists
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Get all sample directories
        try:
            self.samples = [d for d in os.listdir(self.root_dir) 
                            if os.path.isdir(os.path.join(self.root_dir, d))]
            
            # If no samples found, create placeholder synthetic data
            if len(self.samples) == 0:
                print(f"No samples found in {self.root_dir}. Creating synthetic data...")
                # Create minimal synthetic data - just directory structure for now
                for i in range(5):
                    sample_dir = os.path.join(self.root_dir, f"sample_{i}")
                    os.makedirs(sample_dir, exist_ok=True)
                    self.samples.append(f"sample_{i}")
        except FileNotFoundError:
            print(f"Directory not found: {self.root_dir}. Creating it...")
            os.makedirs(self.root_dir, exist_ok=True)
            # Create minimal synthetic data
            for i in range(5):
                sample_dir = os.path.join(self.root_dir, f"sample_{i}")
                os.makedirs(sample_dir, exist_ok=True)
                self.samples.append(f"sample_{i}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: A dictionary containing the sample data.
        """
        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        
        # Load RGB image
        rgb_path = os.path.join(sample_dir, 'rgb.png')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(sample_dir, 'rgb.jpg')
        
        rgb_image = Image.open(rgb_path).convert('RGB')
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # Load mask if available
        mask_path = os.path.join(sample_dir, 'mask.png')
        mask = None
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = transforms.Resize((224, 224))(mask)
            mask = transforms.ToTensor()(mask)
        
        # Create sample dictionary
        sample = {
            'rgb': rgb_image,
            'mask': mask,
            'sample_id': self.samples[idx]
        }
        
        return sample


def build_dataloader(cfg, split='train'):
    """
    Build a dataloader for the dataset.
    
    Args:
        cfg (OmegaConf): Configuration object.
        split (str): 'train' or 'val' split.
        
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    # Check for dataset path and provide fallbacks for different environments
    dataset_path = cfg.data.dataset_path
    
    # Get possible dataset paths
    possible_paths = [
        dataset_path,                              # Original path from config
        os.path.join('/content/dataset'),         # Colab common path
        os.path.join('/content/TripoSR/dataset'), # Colab with TripoSR repo
        './dataset',                              # Relative to current directory
    ]
    
    # Check which path exists
    valid_path = None
    for path in possible_paths:
        if os.path.exists(path):
            valid_path = path
            break
            
    # If no valid path found, create a synthetic dataset
    if valid_path is None:
        print(f"No dataset found at any of the expected locations. Creating a synthetic dataset at '/content/dataset'")
        os.makedirs('/content/dataset/train', exist_ok=True)
        os.makedirs('/content/dataset/val', exist_ok=True)
        # Create minimal synthetic data (empty directories are enough for now)
        for i in range(5):
            os.makedirs(f'/content/dataset/train/sample_{i}', exist_ok=True)
            os.makedirs(f'/content/dataset/val/sample_{i}', exist_ok=True)
        valid_path = '/content/dataset'
    
    print(f"Using dataset at: {valid_path}")
    
    dataset = TripoSRDataset(
        root_dir=valid_path,
        split=split
    )
    
    batch_size = cfg.train.batch_size if split == 'train' else cfg.val.batch_size
    shuffle = split == 'train'
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=split == 'train'
    )
    
    return dataloader
