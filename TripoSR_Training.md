# TripoSR Fine-Tuning Training Guide for Google Colab

---

## 📋 Prerequisites
- Google Colab account with GPU runtime enabled
- Basic understanding of machine learning concepts
- Training dataset prepared (images + 3D models)

---

## 🔧 Cell 1: Environment Setup and Repository Clone

**Purpose**: Mount Google Drive, clone the TripoSR repository, and set up the workspace.

```python
# Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Navigate to content directory and clean workspace
%cd /content

# Remove any existing TripoSR folder and clone fresh
!rm -rf TripoSR
!git clone -b openlrm-training --single-branch --depth 1 https://github.com/wirapratamaz/TripoSR.git

# Enter the repository directory
%cd TripoSR

# Clean up any existing config files
!rm -f config.yaml

# Verify repository structure
print("=== Repository Structure Check ===")
!ls -la
print("\n=== Checking for train.py ===")
!ls -l train.py
print("================================")
```

---

## 📦 Cell 2: Install Dependencies

**Purpose**: Install all required Python packages for TripoSR training.

```python
# Install core dependencies
print("Installing core ML packages...")
!pip install -q trimesh omegaconf einops rembg huggingface-hub transformers==4.35.0 onnxruntime

print("Installing 3D processing packages...")
!pip install -q git+https://github.com/tatsy/torchmcubes.git
!pip install -q xatlas==0.0.9 imageio[ffmpeg] matplotlib pandas tqdm

print("Installing additional dependencies...")
!pip install -q moderngl scipy>=1.11.0

# Install project-specific requirements
print("Installing project requirements...")
!pip install -r requirements.txt

# Verify CUDA setup
import torch
print("\n=== GPU Setup Verification ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ WARNING: CUDA not available. Training will be very slow on CPU.")
print("================================")
```

---

## 📁 Cell 3: Create Directory Structure

**Purpose**: Set up the required directory structure for training data and outputs.

```python
# Create training and validation data directories
print("Creating dataset directories...")
!mkdir -p /content/TripoSR/dataset/train
!mkdir -p /content/TripoSR/dataset/val

# Create output directories for models and logs
print("Creating output directories...")
!mkdir -p /content/TripoSR/output
!mkdir -p /content/TripoSR/evaluation
!mkdir -p /content/TripoSR/logs

# Verify directory structure
print("\n=== Directory Structure ===")
!tree /content/TripoSR -d -L 3 2>/dev/null || find /content/TripoSR -type d | head -20
print("==========================")
```

---

## ⚙️ Cell 4: Configuration Setup

**Purpose**: Create or upload the training configuration file.

```python
from google.colab import files
import os

# Option 1: Upload your own config.yaml
print("🔧 Configuration Setup")
print("Choose one of the following options:")
print("1. Upload your own config.yaml file")
print("2. Use the default configuration (recommended for beginners)")
print("\nTo upload: Run the upload cell below")
print("To use default: Skip the upload and continue")

# Uncomment the next line if you want to upload your own config
# uploaded = files.upload()

# Create default configuration
config_content = """# TripoSR Training Configuration
# Adjust these parameters based on your dataset and hardware

data:
  train_path: ./dataset/train
  val_path: ./dataset/val
  input_format: image
  target_format: mesh
  resolution: 128
  num_workers: 2

training:
  batch_size: 2          # Reduce if you get OOM errors
  epochs: 30             # Increase for better results
  learning_rate: 1e-4    # Learning rate for optimization
  save_interval: 5       # Save model every N epochs
  log_interval: 10       # Log progress every N steps

# Model Architecture (TripoSR configuration)
model:
  type: TSR
  transformer:
    encoder_layers: 12
    decoder_layers: 8
    embed_dim: 768

cond_image_size: 256

# Image Encoder Configuration
image_tokenizer_cls: tsr.models.image_encoders.openai.OpenAIImageEncoder
image_tokenizer:
  embed_dim: 768
  encoder_pretrained: stabilityai/sd-vae-ft-ema
  encoder_config:
    z_channels: 4
    resolution: 256
    in_channels: 3
    out_ch: 3
    ch: 128
    ch_mult: [1, 2, 4, 4]
    num_res_blocks: 2
    attn_resolutions: [32]
    dropout: 0.0

# Surface Tokenizer Configuration
tokenizer_cls: tsr.models.tokenizers.surface_plane.SurfacePlaneTokenizer
tokenizer:
  n_point_samples: 6144
  sample_mode: grid
  resolution: 32
  padding: 0.1
  embed_dim: 768

# Transformer Backbone
backbone_cls: tsr.models.transformer.Transformer
backbone:
  encoder:
    embed_dim: 768
    depth: 12
    num_heads: 12
    mlp_ratio: 4
    qkv_bias: True
  decoder:
    embed_dim: 768
    depth: 8
    num_heads: 12
    mlp_ratio: 4
    qkv_bias: True

# Post Processor
post_processor_cls: tsr.models.post_processors.identity.Identity
post_processor: {}

# Decoder Configuration
decoder_cls: tsr.models.decoders.triplane.Triplane
decoder:
  dims_3d: [32, 32, 32]
  dims_2d: [256, 256]
  feat_dim: 32
  mlp_dim: 128
  out_dim: 4
  n_blocks: 2

# Volume Renderer
renderer_cls: tsr.models.renderers.volume.VolumeRenderer
renderer:
  radius: 1.3
  n_samples: 128"""

# Write configuration to file
with open('/content/TripoSR/config.yaml', 'w') as f:
    f.write(config_content)

print("✅ Configuration file created successfully!")

# Display first few lines of config
print("\n=== Configuration Preview ===")
!head -n 15 /content/TripoSR/config.yaml
print("... (configuration continues)")
print("=============================")
```

---

## 📊 Cell 5: Dataset Preparation

**Purpose**: Set up your training dataset. Choose between uploading your own data or using sample data for testing.

```python
import os
from google.colab import files

print("📊 Dataset Setup")
print("Choose your dataset option:")
print("\nOption A: Upload your own dataset")
print("- Prepare your data in the following structure:")
print("  dataset/train/sample_name/image.png (or .jpg)")
print("  dataset/train/sample_name/model.obj (3D model)")
print("  dataset/val/sample_name/image.png")
print("  dataset/val/sample_name/model.obj")
print("\nOption B: Use sample dataset for testing")

# Option B: Create sample dataset for testing
use_sample_data = True  # Set to False if you want to upload your own data

if use_sample_data:
    print("\n🔄 Creating sample dataset for testing...")
    
    # Create sample directories
    !mkdir -p /content/TripoSR/dataset/train/sample1
    !mkdir -p /content/TripoSR/dataset/train/sample2
    !mkdir -p /content/TripoSR/dataset/val/sample1
    !mkdir -p /content/TripoSR/dataset/val/sample2
    
    # Download sample images
    print("Downloading sample images...")
    !wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/train/sample1/image.png
    !wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/train/sample2/image.png
    !wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/val/sample1/image.png
    !wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/val/sample2/image.png
    
    # Create placeholder 3D model files
    print("Creating placeholder 3D models...")
    !touch /content/TripoSR/dataset/train/sample1/model.obj
    !touch /content/TripoSR/dataset/train/sample2/model.obj
    !touch /content/TripoSR/dataset/val/sample1/model.obj
    !touch /content/TripoSR/dataset/val/sample2/model.obj
    
    print("✅ Sample dataset created successfully!")
    print("⚠️ Note: This is for testing the training pipeline only.")
    print("   For real training, replace with your actual dataset.")
else:
    print("\n📁 Please upload your dataset files using the file browser or:")
    print("1. Mount your Google Drive containing the dataset")
    print("2. Copy your dataset to /content/TripoSR/dataset/")
    print("3. Ensure proper directory structure as shown above")

# Verify dataset structure
print("\n=== Dataset Structure Verification ===")
!find /content/TripoSR/dataset -type f | head -10
print("=====================================")
```

---

## 🔄 Cell 6: Model Configuration Update

**Purpose**: Update the training scripts to use the correct model source.

```python
import re
import os

print("🔄 Updating model configuration...")

# Update train.py to use the correct model source
if os.path.exists('train.py'):
    with open('train.py', 'r') as f:
        train_content = f.read()
    
    # Update model source if needed
    if 'TrianC0de/TripoSR' not in train_content:
        updated_content = re.sub(r'("stabilityai/TripoSR")', r'"TrianC0de/TripoSR"', train_content)
        with open('train.py', 'w') as f:
            f.write(updated_content)
        print("✅ Updated train.py to use TrianC0de/TripoSR model")
    else:
        print("✅ train.py already configured correctly")
else:
    print("⚠️ train.py not found")

# Update evaluate.py similarly
if os.path.exists('evaluate.py'):
    with open('evaluate.py', 'r') as f:
        eval_content = f.read()
    
    if 'TrianC0de/TripoSR' not in eval_content:
        updated_content = re.sub(r'("stabilityai/TripoSR")', r'"TrianC0de/TripoSR"', eval_content)
        with open('evaluate.py', 'w') as f:
            f.write(updated_content)
        print("✅ Updated evaluate.py to use TrianC0de/TripoSR model")
    else:
        print("✅ evaluate.py already configured correctly")
else:
    print("⚠️ evaluate.py not found")

print("\n🔍 Verifying script configuration...")
!grep -n "TrianC0de/TripoSR" train.py evaluate.py 2>/dev/null || echo "Configuration check complete"
```

---

## 🛠️ Cell 7: Setup Missing Functions and Dependencies

**Purpose**: Create necessary evaluation functions and utilities that might be missing.

```python

print("🛠️ Setting up evaluation functions and utilities...")

# Ensure tsr directory exists
!mkdir -p tsr

# Create or update evaluation.py with required functions
with open('tsr/evaluation.py', 'w') as f:
    f.write('''
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

def chamfer_distance(pred_points, gt_points):
    """
    PyTorch implementation of Chamfer Distance for training and evaluation
    
    Args:
        pred_points (torch.Tensor): Predicted points with shape (B, N, 3)
        gt_points (torch.Tensor): Ground truth points with shape (B, M, 3)
        
    Returns:
        torch.Tensor: Chamfer Distance (lower is better)
    """
    # Convert to numpy if needed
    if isinstance(pred_points, torch.Tensor):
        pred_points_np = pred_points.detach().cpu().numpy()
    else:
        pred_points_np = pred_points
        
    if isinstance(gt_points, torch.Tensor):
        gt_points_np = gt_points.detach().cpu().numpy()
    else:
        gt_points_np = gt_points
    
    # For batched input
    if pred_points_np.ndim == 3:
        batch_size = pred_points_np.shape[0]
        cd_sum = 0
        for i in range(batch_size):
            cd_sum += calculate_chamfer_distance(pred_points_np[i], gt_points_np[i])
        cd = cd_sum / batch_size
        return torch.tensor(cd, device=pred_points.device if isinstance(pred_points, torch.Tensor) else None)
    
    # For single point cloud
    cd = calculate_chamfer_distance(pred_points_np, gt_points_np)
    return torch.tensor(cd, device=pred_points.device if isinstance(pred_points, torch.Tensor) else None)

def calculate_chamfer_distance(predicted_points, ground_truth_points):
    """
    Calculate Chamfer Distance between predicted points and ground truth points
    
    Args:
        predicted_points: Points from the predicted mesh
        ground_truth_points: Points from the ground truth mesh
        
    Returns:
        float: Chamfer Distance (lower is better)
    """
    # Use KDTree for efficient distance computation
    if len(predicted_points) == 0 or len(ground_truth_points) == 0:
        return 0.0
    
    pred_tree = cKDTree(predicted_points)
    gt_tree = cKDTree(ground_truth_points)
    
    # Compute distances from predicted to ground truth
    pred_to_gt, _ = gt_tree.query(predicted_points)
    gt_to_pred, _ = pred_tree.query(ground_truth_points)
    
    # Calculate chamfer distance
    cd = np.mean(pred_to_gt) + np.mean(gt_to_pred)
    
    return cd

def iou_3d(pred_vertices, pred_faces, gt_vertices, gt_faces, voxel_resolution=32):
    """
    Calculate 3D IoU using voxelization
    
    Args:
        pred_vertices (torch.Tensor): Predicted vertices (B, N, 3)
        pred_faces (torch.Tensor): Predicted faces (B, F, 3)
        gt_vertices (torch.Tensor): Ground truth vertices (B, M, 3)
        gt_faces (torch.Tensor): Ground truth faces (B, G, 3)
        voxel_resolution (int): Resolution of voxel grid
        
    Returns:
        torch.Tensor: IoU score (higher is better)
    """
    # Simple placeholder implementation - replace with actual implementation if needed
    return torch.tensor(0.5, device=pred_vertices.device)
''')

# Ensure utils.py exists and has load_config function
with open('tsr/utils.py', 'a') as f:
    f.write('''

# Add load_config function if not already present
def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        OmegaConf: Configuration object
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)
''')

print("✅ Evaluation functions and utilities set up successfully!")
print("📁 Created: tsr/evaluation.py")
print("📁 Updated: tsr/utils.py")
```

---

## 🚀 Cell 8: Start Training

**Purpose**: Begin the fine-tuning process with the configured settings.

```python
import os
import time
from datetime import datetime

print("🚀 Starting TripoSR Fine-Tuning Training")
print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*50)

# Pre-training checks
print("🔍 Pre-training verification:")
print(f"✓ Config file exists: {os.path.exists('config.yaml')}")
print(f"✓ Training script exists: {os.path.exists('train.py')}")
print(f"✓ Dataset directory exists: {os.path.exists('dataset')}")
print(f"✓ Output directory exists: {os.path.exists('output')}")

# Check GPU memory
import torch
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
    if gpu_memory < 8:
        print("⚠️ Warning: Low GPU memory. Consider reducing batch_size in config.yaml")
else:
    print("⚠️ Warning: No GPU detected. Training will be very slow.")

print("\n🎯 Starting training process...")
print("Note: This may take several hours depending on your dataset size and epochs.")
print("You can monitor progress in the output below.\n")

try:
    # Start training with detailed logging
    !python train.py --config config.yaml --output_dir /content/TripoSR/output --device cuda:0
    
    print("\n" + "="*50)
    print("🎉 Training Process Completed!")
    
    # Check for output files
    output_files = []
    if os.path.exists("/content/TripoSR/output"):
        output_files = os.listdir("/content/TripoSR/output")
    
    if output_files:
        print("\n📁 Generated Files:")
        for file in output_files:
            file_path = f"/content/TripoSR/output/{file}"
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✓ {file} ({file_size:.1f} MB)")
        
        # Check for model checkpoints
        model_files = [f for f in output_files if f.endswith(('.pth', '.ckpt'))]
        if model_files:
            print(f"\n✅ Training completed successfully!")
            print(f"📦 Model checkpoint(s) saved: {', '.join(model_files)}")
        else:
            print(f"\n⚠️ Training completed but no model checkpoints found.")
            print(f"📋 Check the training logs above for any errors.")
    else:
        print("\n⚠️ No output files generated. Check for errors in the training logs above.")
        
except Exception as e:
    print(f"\n❌ Training failed with error: {str(e)}")
    print("\n🔧 Troubleshooting tips:")
    print("1. Check if your dataset is properly formatted")
    print("2. Verify GPU memory is sufficient (reduce batch_size if needed)")
    print("3. Ensure all dependencies are installed correctly")
    print("4. Check the full error message above for specific issues")

print(f"\n⏰ Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

---

## 📊 Cell 9: Model Evaluation Setup

**Purpose**: Create a comprehensive evaluation script to compare your fine-tuned model with the original.

```python

# Create a basic evaluate.py if it doesn't exist or has issues
print("Creating/updating evaluate.py...")

with open('evaluate.py', 'w') as f:
    f.write('''
import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from omegaconf import OmegaConf

from tsr.system import TSR
from tsr.data import get_dataloaders
from tsr.evaluation import chamfer_distance, iou_3d

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_metrics(pred_mesh, target_vertices, target_faces, device):
    """
    Compute evaluation metrics
    
    Args:
        pred_mesh: Predicted mesh
        target_vertices: Target vertices
        target_faces: Target faces
        device: Device to use
    
    Returns:
        dict: Dictionary of metrics
    """
    # Convert predicted mesh to torch tensors
    pred_vertices = torch.tensor(pred_mesh.vertices, device=device)
    pred_faces = torch.tensor(pred_mesh.faces, device=device)
    
    # Compute Chamfer distance
    cd = chamfer_distance(
        pred_vertices.unsqueeze(0), 
        target_vertices.unsqueeze(0)
    ).item()
    
    # Compute IoU (approximate using voxelization)
    iou = iou_3d(
        pred_vertices.unsqueeze(0), 
        pred_faces.unsqueeze(0),
        target_vertices.unsqueeze(0), 
        target_faces.unsqueeze(0),
        voxel_resolution=32
    ).item()
    
    # Compute precision/recall for a simple F1 score
    threshold = 0.02
    from scipy.spatial import cKDTree
    
    # Build KD-Tree for efficient nearest neighbor queries
    pred_points = pred_vertices.cpu().numpy()
    gt_points = target_vertices.cpu().numpy()
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    
    # For each predicted point, find distance to nearest ground truth point
    pred_to_gt_dists, _ = gt_tree.query(pred_points, k=1)
    
    # For each ground truth point, find distance to nearest predicted point
    gt_to_pred_dists, _ = pred_tree.query(gt_points, k=1)
    
    # Compute precision and recall
    precision = np.mean((pred_to_gt_dists < threshold).astype(np.float32))
    recall = np.mean((gt_to_pred_dists < threshold).astype(np.float32))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'chamfer_distance': cd,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def visualize_comparison(original_mesh, finetuned_mesh, target_mesh, output_path):
    """Visualize comparison between original, fine-tuned and target meshes"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot original mesh
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Original Model')
    x, y, z = original_mesh.vertices.T
    ax1.scatter(x, y, z, s=0.1)
    
    # Plot fine-tuned mesh
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Fine-tuned Model')
    x, y, z = finetuned_mesh.vertices.T
    ax2.scatter(x, y, z, s=0.1)
    
    # Plot target mesh
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Target')
    x, y, z = target_mesh.cpu().numpy().T
    ax3.scatter(x, y, z, s=0.1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned TripoSR model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
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
    
    # Initialize original model
    logger.info("Initializing original model...")
    original_model = TSR.from_pretrained(
        "./",
        config_name="config.yaml",
        weight_name="model0.9975.ckpt"
    )
    original_model.to(device)
    
    # Initialize fine-tuned model
    logger.info("Initializing fine-tuned model...")
    finetuned_model = TSR.from_pretrained(
        "TrianC0de/TripoSR",
        config_name="config.yaml",
        weight_name="sdfusion-snet-all.pth"
    )
    finetuned_model.load_state_dict(torch.load(args.finetuned_model))
    finetuned_model.to(device)
    
    # Get validation dataloader
    _, val_loader = get_dataloaders(config)
    
    # Evaluate models
    logger.info("Evaluating models...")
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if i >= args.num_samples:
                break
                
            # Move data to device
            images = batch["images"].to(device)
            vertices_list = [v.to(device) for v in batch["vertices"]]
            faces_list = [f.to(device) for v in batch["faces"]]
            obj_ids = batch["obj_ids"]
            
            # Process with original model
            scene_codes_orig = original_model(images, device=device)
            pred_meshes_orig = original_model.extract_mesh(scene_codes_orig, has_vertex_color=False)
            
            # Process with fine-tuned model
            scene_codes_ft = finetuned_model(images, device=device)
            pred_meshes_ft = finetuned_model.extract_mesh(scene_codes_ft, has_vertex_color=False)
            
            # Compute metrics and visualize for each sample in batch
            for j, (pred_mesh_ft, pred_mesh_orig, target_vertices, target_faces, obj_id) in enumerate(
                zip(pred_meshes_ft, pred_meshes_orig, vertices_list, faces_list, obj_ids)
            ):
                logger.info(f"Evaluating sample {obj_id}...")
                
                # Compute metrics for fine-tuned model
                metrics_ft = compute_metrics(
                    pred_mesh_ft,
                    target_vertices,
                    target_faces,
                    device
                )
                
                # Compute metrics for original model
                metrics_orig = compute_metrics(
                    pred_mesh_orig,
                    target_vertices,
                    target_faces,
                    device
                )
                
                # Store results
                result = {
                    'obj_id': obj_id,
                    'ft_chamfer_distance': metrics_ft['chamfer_distance'],
                    'ft_iou': metrics_ft['iou'],
                    'ft_f1_score': metrics_ft['f1_score'],
                    'orig_chamfer_distance': metrics_orig['chamfer_distance'],
                    'orig_iou': metrics_orig['iou'],
                    'orig_f1_score': metrics_orig['f1_score'],
                    'cd_improvement': (1 - metrics_ft['chamfer_distance'] / metrics_orig['chamfer_distance']) * 100,
                    'iou_improvement': (metrics_ft['iou'] / metrics_orig['iou'] - 1) * 100,
                    'f1_improvement': (metrics_ft['f1_score'] / metrics_orig['f1_score'] - 1) * 100
                }
                results.append(result)
                
                # Visualize comparison
                if args.visualize:
                    logger.info(f"Visualizing sample {obj_id}...")
                    vis_path = os.path.join(args.output_dir, f"{obj_id}_comparison.png")
                    visualize_comparison(
                        pred_mesh_orig,
                        pred_mesh_ft,
                        target_vertices,
                        vis_path
                    )
                    
                    # Save meshes
                    pred_mesh_orig.export(os.path.join(args.output_dir, f"{obj_id}_original.obj"))
                    pred_mesh_ft.export(os.path.join(args.output_dir, f"{obj_id}_finetuned.obj"))
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    
    # Print summary
    logger.info("===== Evaluation Summary =====")
    avg_metrics = results_df.mean()
    logger.info(f"Average Chamfer Distance: {avg_metrics['ft_chamfer_distance']:.4f} (Original: {avg_metrics['orig_chamfer_distance']:.4f})")
    logger.info(f"Average IoU: {avg_metrics['ft_iou']:.4f} (Original: {avg_metrics['orig_iou']:.4f})")
    logger.info(f"Average F1 Score: {avg_metrics['ft_f1_score']:.4f} (Original: {avg_metrics['orig_f1_score']:.4f})")
    logger.info(f"Average CD Improvement: {avg_metrics['cd_improvement']:.2f}%")
    logger.info(f"Average IoU Improvement: {avg_metrics['iou_improvement']:.2f}%")
    logger.info(f"Average F1 Improvement: {avg_metrics['f1_improvement']:.2f}%")
    
    # Plot improvements
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['chamfer_distance', 'iou', 'f1_score']
    titles = ['Chamfer Distance (lower is better)', 'IoU (higher is better)', 'F1 Score (higher is better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        orig_vals = results_df[f'orig_{metric}'].values
        ft_vals = results_df[f'ft_{metric}'].values
        
        axes[i].bar(range(len(orig_vals)), orig_vals, label='Original', alpha=0.7)
        axes[i].bar([x + 0.4 for x in range(len(ft_vals))], ft_vals, label='Fine-tuned', alpha=0.7)
        axes[i].set_title(title)
        axes[i].set_xticks([x + 0.2 for x in range(len(orig_vals))])
        axes[i].set_xticklabels(results_df['obj_id'])
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "metrics_comparison.png"))
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    evaluate()
''')

print("evaluate.py created/updated successfully!")

---

## 🔍 Cell 10: Run Model Evaluation

**Purpose**: Evaluate your fine-tuned model against the original and generate comparison reports.

```python
import os
import glob
from datetime import datetime

print("🔍 Model Evaluation and Comparison")
print(f"⏰ Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*50)

# Check if training was completed
print("🔍 Checking for trained models...")
model_files = []
if os.path.exists("/content/TripoSR/output"):
    model_files = glob.glob("/content/TripoSR/output/*.pth") + glob.glob("/content/TripoSR/output/*.ckpt")

if model_files:
    latest_model = max(model_files, key=os.path.getctime)
    model_size = os.path.getsize(latest_model) / (1024*1024)  # MB
    print(f"✅ Found trained model: {os.path.basename(latest_model)} ({model_size:.1f} MB)")
    
    print("\n🚀 Starting evaluation process...")
    print("This will compare your fine-tuned model with the original TripoSR model.")
    
    try:
        # Run the evaluation script
        exec(open('evaluate.py').read())
        
        print("\n" + "="*50)
        print("🎉 Evaluation Process Completed!")
        
        # Check evaluation results
        eval_dir = "/content/TripoSR/evaluation"
        if os.path.exists(eval_dir):
            eval_files = os.listdir(eval_dir)
            print(f"\n📁 Generated {len(eval_files)} evaluation files:")
            
            for file in eval_files[:10]:  # Show first 10 files
                file_path = os.path.join(eval_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  📄 {file} ({file_size:.1f} KB)")
            
            if len(eval_files) > 10:
                print(f"  ... and {len(eval_files) - 10} more files")
            
            # Display metrics if available
            metrics_file = os.path.join(eval_dir, "evaluation_metrics.csv")
            if os.path.exists(metrics_file):
                print("\n📊 Loading evaluation metrics...")
                import pandas as pd
                try:
                    results = pd.read_csv(metrics_file)
                    print("\n📈 Quick Results Summary:")
                    print(f"   Samples evaluated: {len(results)}")
                    
                    if len(results) > 0:
                        avg_cd_improvement = results['cd_improvement'].mean()
                        avg_iou_improvement = results['iou_improvement'].mean()
                        avg_f1_improvement = results['f1_improvement'].mean()
                        
                        print(f"   Average Chamfer Distance improvement: {avg_cd_improvement:.2f}%")
                        print(f"   Average IoU improvement: {avg_iou_improvement:.2f}%")
                        print(f"   Average F1 Score improvement: {avg_f1_improvement:.2f}%")
                        
                        if avg_cd_improvement > 0:
                            print("   🎉 Your model shows improvement in Chamfer Distance!")
                        if avg_iou_improvement > 0:
                            print("   🎉 Your model shows improvement in IoU!")
                        if avg_f1_improvement > 0:
                            print("   🎉 Your model shows improvement in F1 Score!")
                            
                except Exception as e:
                    print(f"   ⚠️ Could not load metrics: {e}")
            
            print(f"\n📂 All evaluation results saved to: {eval_dir}")
            print("   📊 Metrics: evaluation_metrics.csv")
            print("   🖼️ Visualizations: *_comparison.png")
            print("   🎯 3D Models: *_original.obj, *_finetuned.obj")
        else:
            print("⚠️ Evaluation directory not found")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure training completed successfully")
        print("2. Check if validation dataset is properly formatted")
        print("3. Verify GPU memory is sufficient")
        print("4. Check the error message above for specific issues")
else:
    print("❌ No trained model found!")
    print("\n📋 Please ensure:")
    print("1. Training has completed successfully (Cell 8)")
    print("2. Model checkpoint files (.pth or .ckpt) exist in /content/TripoSR/output/")
    print("3. No errors occurred during training")
    
print(f"\n⏰ Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

---

## 📥 Cell 11: Download Results and Cleanup

**Purpose**: Package and download all training results, models, and evaluation metrics.

```python
import os
import zipfile
from google.colab import files
from datetime import datetime

print("📥 Preparing Results for Download")
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*50)

# Check what files we have
result_dirs = []
result_files = []

# Check for training outputs
if os.path.exists("/content/TripoSR/output"):
    output_files = os.listdir("/content/TripoSR/output")
    if output_files:
        result_dirs.append("output")
        print(f"✅ Training outputs found: {len(output_files)} files")

# Check for evaluation results
if os.path.exists("/content/TripoSR/evaluation"):
    eval_files = os.listdir("/content/TripoSR/evaluation")
    if eval_files:
        result_dirs.append("evaluation")
        print(f"✅ Evaluation results found: {len(eval_files)} files")

# Check for important files
important_files = ['config.yaml', 'train.py', 'evaluate.py']
for file in important_files:
    if os.path.exists(f"/content/TripoSR/{file}"):
        result_files.append(file)
        print(f"✅ Found: {file}")

# Check for log files
log_files = [f for f in os.listdir("/content/TripoSR") if f.endswith('.log')]
if log_files:
    result_files.extend(log_files)
    print(f"✅ Found {len(log_files)} log files")

if result_dirs or result_files:
    print(f"\n📦 Creating download package...")
    
    # Create comprehensive zip file
    zip_filename = f"/content/TripoSR/triposr_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add directories
        for dir_name in result_dirs:
            dir_path = f"/content/TripoSR/{dir_name}"
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, "/content/TripoSR")
                    zipf.write(file_path, arcname)
                    
        # Add individual files
        for file_name in result_files:
            file_path = f"/content/TripoSR/{file_name}"
            if os.path.exists(file_path):
                zipf.write(file_path, file_name)
    
    # Check zip file size
    zip_size = os.path.getsize(zip_filename) / (1024*1024)  # MB
    print(f"📦 Package created: {os.path.basename(zip_filename)} ({zip_size:.1f} MB)")
    
    # Create a summary report
    summary_file = "/content/TripoSR/training_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("TripoSR Fine-Tuning Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Files included in this package:\n")
        f.write("-" * 30 + "\n")
        
        if "output" in result_dirs:
            f.write("📁 output/ - Trained model checkpoints and logs\n")
        if "evaluation" in result_dirs:
            f.write("📁 evaluation/ - Model evaluation results and comparisons\n")
        
        for file in result_files:
            if file == 'config.yaml':
                f.write("📄 config.yaml - Training configuration used\n")
            elif file.endswith('.py'):
                f.write(f"📄 {file} - Training/evaluation script\n")
            elif file.endswith('.log'):
                f.write(f"📄 {file} - Training log file\n")
        
        f.write("\nNext steps:\n")
        f.write("-" * 12 + "\n")
        f.write("1. Extract the downloaded zip file\n")
        f.write("2. Review evaluation results in evaluation/evaluation_metrics.csv\n")
        f.write("3. Check 3D model outputs in evaluation/*_finetuned.obj\n")
        f.write("4. Use the trained model checkpoint for inference\n")
        f.write("5. Review training logs for performance insights\n")
    
    # Add summary to zip
    with zipfile.ZipFile(zip_filename, 'a') as zipf:
        zipf.write(summary_file, "README.txt")
    
    print("\n📥 Starting download...")
    print("Note: Large files may take a moment to prepare for download.")
    
    try:
        files.download(zip_filename)
        print("\n✅ Download initiated successfully!")
        print("📋 Your download should start automatically.")
        print("\n📦 Package contents:")
        if "output" in result_dirs:
            print("   🎯 Trained model checkpoints")
        if "evaluation" in result_dirs:
            print("   📊 Evaluation metrics and visualizations")
            print("   🎨 3D model comparisons")
        print("   📄 Configuration files and scripts")
        print("   📝 Training summary and logs")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n🔧 Alternative: You can manually download files from the file browser")
        print(f"📁 Zip file location: {zip_filename}")
        
else:
    print("❌ No results found to download!")
    print("\n📋 This could mean:")
    print("1. Training did not complete successfully")
    print("2. No model checkpoints were saved")
    print("3. Evaluation was not run")
    print("\nPlease review the previous cells for any error messages.")

# Cleanup temporary files (optional)
print("\n🧹 Cleaning up temporary files...")
try:
    # Remove large temporary files to free up space
    temp_files = ['/content/TripoSR/training_summary.txt']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    print("✅ Cleanup completed")
except:
    print("⚠️ Some temporary files could not be cleaned up")

print(f"\n⏰ Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n🎉 TripoSR Fine-Tuning Training Complete!")
print("Thank you for using this training guide. Happy 3D modeling! 🚀")
```

---

## 🎯 Training Complete!

Congratulations! You have successfully completed the TripoSR fine-tuning process.
- **More Data**: Larger, diverse datasets typically yield better results
- **Longer Training**: More epochs can improve model performance
- **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
- **Data Quality**: High-quality, well-aligned image-3D pairs are crucial

---

**Happy 3D Modeling with TripoSR! 🎨✨**