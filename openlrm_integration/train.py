#!/usr/bin/env python
"""
Main training script for TripoSR with OpenLRM integration.
"""
import os
import argparse
import torch
from omegaconf import OmegaConf

from openlrm.utils.logging import configure_logger, get_logger
# Add directory to path to ensure imports work in Colab
import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now try to import
try:
    from trainers.openlrm_trainer import TripoSRTrainer
except ImportError:
    # Fallback to absolute path
    trainer_path = os.path.join(current_dir, "trainers")
    if trainer_path not in sys.path:
        sys.path.append(trainer_path)
    from openlrm_trainer import TripoSRTrainer
from utils.utils import setup_experiment_directories, convert_to_ckpt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train TripoSR with OpenLRM")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--export_ckpt', action='store_true',
                        help='Export the model to a .ckpt file after training')
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logger
    configure_logger()
    logger = get_logger(__name__)
    
    logger.info(f"Using configuration file: {args.config}")
    
    # Setup experiment directories
    cfg = OmegaConf.load(args.config)
    cfg = setup_experiment_directories(cfg)
    
    # Log configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Train model
    logger.info("Starting training...")
    with TripoSRTrainer() as trainer:
        trainer.run()
        
        # Export model to .ckpt if requested
        if args.export_ckpt:
            logger.info("Exporting model to .ckpt format...")
            safetensors_path = os.path.join(
                cfg.saver.model_root,
                cfg.experiment.parent, cfg.experiment.child,
                f"model_{trainer.global_step:06d}.safetensors"
            )
            ckpt_path = convert_to_ckpt(safetensors_path)
            logger.info(f"Model exported to {ckpt_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
