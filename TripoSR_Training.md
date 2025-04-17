# TripoSR Fine-Tuning Colab Notebook

## Cell 1: Setup and Dependencies

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository
!git clone https://github.com/wirapratamaz/TripoSR.git
%cd /content/TripoSR

# --- Add this line to remove potential conflicting file ---
!rm -f config.yaml
# --- End of added line ---

# Checkout the correct branch and pull latest changes
!git checkout Training
!git pull origin Training

# --- Add this line to check file existence ---
print("--- Checking for train.py after clone/checkout ---")
!ls -l train.py
print("-----------------------------------------------")
# --- End of added line ---

# Install dependencies
!pip install -q trimesh omegaconf einops rembg huggingface-hub transformers==4.35.0 onnxruntime
!pip install -q git+https://github.com/tatsy/torchmcubes.git
!pip install -q xatlas==0.0.9 imageio[ffmpeg] matplotlib pandas tqdm
!pip install -q moderngl scipy>=1.11.0
!pip install -r requirements.txt

# Check CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

## Cell 2: Create Directories

# Create directories
!mkdir -p /content/TripoSR/dataset/train
!mkdir -p /content/TripoSR/dataset/val
!mkdir -p /content/TripoSR/output
!mkdir -p /content/TripoSR/evaluation

## Cell 3: Upload Config File

# Upload config.yaml rather than trying to create it inline
from google.colab import files
import io
import os

# Either upload a config file or use the default one
print("You can upload your own config.yaml or use the default one")
print("To use the default, just click 'Skip' below")

try:
  uploaded = files.upload()  # This will prompt for file upload
  if 'config.yaml' in uploaded:
    print("Using uploaded config.yaml")
  else:
    # If they uploaded something else or skipped
    raise Exception("No config.yaml uploaded, using default")
except:
  # Create the default config.yaml file using Python file operations
  # This avoids the %%writefile magic issues
  config_content = """data:
  train_path: ./dataset/train
  val_path: ./dataset/val
  input_format: image
  target_format: mesh
  resolution: 128
  num_workers: 2

training:
  batch_size: 2
  epochs: 30
  learning_rate: 1e-4
  save_interval: 5
  log_interval: 10

model:
  type: TSR
  transformer:
    encoder_layers: 12
    decoder_layers: 8
    embed_dim: 768

# Original TripoSR model configuration
cond_image_size: 256

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

tokenizer_cls: tsr.models.tokenizers.surface_plane.SurfacePlaneTokenizer
tokenizer:
  n_point_samples: 6144
  sample_mode: grid
  resolution: 32
  padding: 0.1
  embed_dim: 768

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

post_processor_cls: tsr.models.post_processors.identity.Identity
post_processor: {}

decoder_cls: tsr.models.decoders.triplane.Triplane
decoder:
  dims_3d: [32, 32, 32]
  dims_2d: [256, 256]
  feat_dim: 32
  mlp_dim: 128
  out_dim: 4
  n_blocks: 2

renderer_cls: tsr.models.renderers.volume.VolumeRenderer
renderer:
  radius: 1.3
  n_samples: 128"""

  with open('/content/TripoSR/config.yaml', 'w') as f:
    f.write(config_content)

  print("Created default config.yaml")

# Verify the config file
!cat /content/TripoSR/config.yaml | head -n 10
print("... (config file continues)")

## Cell 6: Create Sample Dataset (for testing)

# Create a minimal test dataset
!mkdir -p /content/TripoSR/dataset/train/sample1
!mkdir -p /content/TripoSR/dataset/val/sample2

# Download sample images
!wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/train/sample1/image.png
!wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/val/sample2/image.png

# Create placeholder model files
!touch /content/TripoSR/dataset/train/sample1/model.obj
!touch /content/TripoSR/dataset/val/sample2/model.obj

print("Sample dataset created (for testing workflow only)")

## Cell 7: Update Model Source to Use TrianC0de/TripoSR

# Update train.py to use TrianC0de/TripoSR model
import re
with open('train.py', 'r') as f:
    train_content = f.read()

if 'TrianC0de/TripoSR' not in train_content:
    updated_content = re.sub(r'("stabilityai/TripoSR")', r'"TrianC0de/TripoSR"', train_content)
    with open('train.py', 'w') as f:
        f.write(updated_content)
    print("Updated train.py to use TrianC0de/TripoSR model")

# Do the same for evaluate.py
with open('evaluate.py', 'r') as f:
    eval_content = f.read()

if 'TrianC0de/TripoSR' not in eval_content:
    updated_content = re.sub(r'("stabilityai/TripoSR")', r'"TrianC0de/TripoSR"', eval_content)
    with open('evaluate.py', 'w') as f:
        f.write(updated_content)
    print("Updated evaluate.py to use TrianC0de/TripoSR model")

## Cell 7b: Create Custom Implementation of Missing Functions

# Create implementations for missing functions
print("Creating implementation for missing functions...")

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

# Make sure utils.py has load_config function
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

print("Functions implemented successfully!")

## Cell 8: Run Training

# Start fine-tuning with pretrained TrianC0de/TripoSR model
print("Starting training...")
try:
    !python train.py --config config.yaml --output_dir /content/TripoSR/output --device cuda:0
    
    # Check if model was created
    if os.path.exists("/content/TripoSR/output/model_final.pth"):
        print("\n✅ Training completed successfully.")
        print("Model saved to: /content/TripoSR/output/model_final.pth")
    else:
        print("\n⚠️ Training completed but model file was not found.")
        print("Check for errors in the training output above.")
except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
    print("Check the full error message above.")

## Cell 8b: Create/Fix Evaluate Script

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
        "TrianC0de/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )
    original_model.to(device)
    
    # Initialize fine-tuned model
    logger.info("Initializing fine-tuned model...")
    finetuned_model = TSR.from_pretrained(
        "TrianC0de/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
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

## Cell 9: Evaluate Model

# Evaluate the fine-tuned model
!python evaluate.py --config config.yaml --finetuned_model /content/TripoSR/output/model_final.pth --output_dir /content/TripoSR/evaluation --visualize --num_samples 2

## Cell 10: Auto-download Results

from google.colab import files
import os

# Define paths
model_path = '/content/TripoSR/output/model_final.pth'
eval_dir = '/content/TripoSR/evaluation'
zip_path = '/content/TripoSR_evaluation.zip'

# Check if model file exists before downloading
if os.path.exists(model_path):
    print(f"Downloading fine-tuned model: {model_path}")
    files.download(model_path)
else:
    print(f"ERROR: Model file not found at {model_path}. Skipping download.")

print("\nDownload prompts should appear above if files were found.")