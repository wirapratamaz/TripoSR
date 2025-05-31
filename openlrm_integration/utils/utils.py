"""
Utility functions for OpenLRM integration with TripoSR.
"""
import os
import torch
import numpy as np
import safetensors.torch
from omegaconf import OmegaConf


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        OmegaConf: Configuration object.
    """
    cfg = OmegaConf.load(config_path)
    return cfg


def save_config(cfg, output_path):
    """
    Save configuration to a YAML file.
    
    Args:
        cfg (OmegaConf): Configuration object.
        output_path (str): Path to save the configuration file.
    """
    with open(output_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


def convert_to_ckpt(safetensors_path, output_path=None):
    """
    Convert a safetensors model to a PyTorch checkpoint file.
    
    Args:
        safetensors_path (str): Path to the safetensors model.
        output_path (str, optional): Path to save the checkpoint file. If None, use the same path with .ckpt extension.
        
    Returns:
        str: Path to the saved checkpoint file.
    """
    # Try multiple possible paths for the safetensors file
    possible_paths = [
        safetensors_path,  # Original path provided
    ]
    
    # Extract step number from the original path if it contains it
    step_str = None
    import re
    step_match = re.search(r'(\d{6})', os.path.basename(safetensors_path))
    if step_match:
        step_str = step_match.group(1)
    
    # If we have a step number, try additional path patterns
    if step_str:
        base_dir = os.path.dirname(safetensors_path)
        checkpoint_style_path = os.path.join(base_dir.replace('models', 'checkpoints'), step_str, 'model.safetensors')
        possible_paths.append(checkpoint_style_path)
        
        # Try looking directly in the checkpoints directory
        exp_parts = base_dir.split('/')
        if len(exp_parts) >= 3:
            parent, child = exp_parts[-2], exp_parts[-1]
            checkpoint_alt_path = f"./checkpoints/{parent}/{child}/{step_str}/model.safetensors"
            possible_paths.append(checkpoint_alt_path)
    
    # Try all possible paths
    for path in possible_paths:
        print(f"Trying to load safetensors from: {path}")
        if os.path.exists(path):
            print(f"Found safetensors file at: {path}")
            if output_path is None:
                # Use the originally requested path for output, even if input was found elsewhere
                output_path = os.path.splitext(safetensors_path)[0] + '.ckpt'
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load model from safetensors
            state_dict = safetensors.torch.load_file(path)
            
            # Save as PyTorch checkpoint
            torch.save({'model': state_dict}, output_path)
            
            print(f"Saved checkpoint to: {output_path}")
            return output_path
    
    # If we get here, no valid path was found
    raise FileNotFoundError(f"Could not find safetensors file at any of the following paths: {possible_paths}")



def setup_experiment_directories(cfg):
    """
    Setup experiment directories for logging, checkpoints, etc.
    
    Args:
        cfg (OmegaConf): Configuration object.
        
    Returns:
        OmegaConf: Updated configuration object.
    """
    # Create experiment name
    experiment_name = f"{cfg.experiment.parent}/{cfg.experiment.child}"
    
    # Create directories
    os.makedirs(os.path.join(cfg.logger.log_root, cfg.experiment.parent, cfg.experiment.child), exist_ok=True)
    os.makedirs(os.path.join(cfg.logger.tracker_root, cfg.experiment.parent, cfg.experiment.child), exist_ok=True)
    os.makedirs(os.path.join(cfg.saver.checkpoint_root, cfg.experiment.parent, cfg.experiment.child), exist_ok=True)
    os.makedirs(os.path.join(cfg.saver.model_root, cfg.experiment.parent, cfg.experiment.child), exist_ok=True)
    
    return cfg
