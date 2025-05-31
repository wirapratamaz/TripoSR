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
    if output_path is None:
        output_path = os.path.splitext(safetensors_path)[0] + '.ckpt'
    
    # Load model from safetensors
    state_dict = safetensors.torch.load_file(safetensors_path)
    
    # Save as PyTorch checkpoint
    torch.save({'model': state_dict}, output_path)
    
    return output_path


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
