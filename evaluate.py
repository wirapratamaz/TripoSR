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
    
    # Compute F1 score (derived from precision and recall)
    # Here we use a threshold based on chamfer distance
    threshold = 0.02
    precision, recall = compute_precision_recall(
        pred_vertices.cpu().numpy(),
        target_vertices.cpu().numpy(),
        threshold
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'chamfer_distance': cd,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def compute_precision_recall(pred_points, gt_points, threshold):
    """
    Compute precision and recall for point clouds
    
    Args:
        pred_points: Predicted points
        gt_points: Ground truth points
        threshold: Distance threshold
    
    Returns:
        tuple: (precision, recall)
    """
    from scipy.spatial import cKDTree
    
    # Build KD-Tree for efficient nearest neighbor queries
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    
    # For each predicted point, find distance to nearest ground truth point
    pred_to_gt_dists, _ = gt_tree.query(pred_points, k=1)
    
    # For each ground truth point, find distance to nearest predicted point
    gt_to_pred_dists, _ = pred_tree.query(gt_points, k=1)
    
    # Compute precision and recall
    precision = np.mean((pred_to_gt_dists < threshold).astype(np.float32))
    recall = np.mean((gt_to_pred_dists < threshold).astype(np.float32))
    
    return precision, recall


def visualize_comparison(image, pred_mesh_ft, pred_mesh_orig, target_vertices, target_faces, metrics_ft, metrics_orig, obj_id, save_dir):
    """
    Visualize comparison between fine-tuned model, pretrained model and ground truth
    
    Args:
        image: Input image
        pred_mesh_ft: Predicted mesh from fine-tuned model
        pred_mesh_orig: Predicted mesh from original model
        target_vertices: Target vertices
        target_faces: Target faces
        metrics_ft: Metrics from fine-tuned model
        metrics_orig: Metrics from original model
        obj_id: Object ID
        save_dir: Directory to save visualizations
    """
    import trimesh
    from trimesh.viewer import scene_to_png
    
    # Create a figure with multiple plots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot input image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Create a scene with the target mesh
    target_mesh = trimesh.Trimesh(vertices=target_vertices.cpu().numpy(), 
                                 faces=target_faces.cpu().numpy())
    
    # Plot target mesh
    target_scene = trimesh.Scene(target_mesh)
    target_png = scene_to_png(
        target_scene,
        resolution=(256, 256),
        background=[255, 255, 255, 255]
    )
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(target_png)
    ax2.set_title("Ground Truth")
    ax2.axis('off')
    
    # Plot fine-tuned model prediction
    ft_scene = trimesh.Scene(pred_mesh_ft)
    ft_png = scene_to_png(
        ft_scene,
        resolution=(256, 256),
        background=[255, 255, 255, 255]
    )
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(ft_png)
    ax3.set_title("Fine-tuned Model")
    ax3.axis('off')
    
    # Plot original model prediction
    orig_scene = trimesh.Scene(pred_mesh_orig)
    orig_png = scene_to_png(
        orig_scene,
        resolution=(256, 256),
        background=[255, 255, 255, 255]
    )
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(orig_png)
    ax4.set_title("Original Model")
    ax4.axis('off')
    
    # Plot comparison metrics
    ax5 = fig.add_subplot(2, 3, 5)
    metrics = ['chamfer_distance', 'iou', 'f1_score']
    ft_values = [metrics_ft[m] for m in metrics]
    orig_values = [metrics_orig[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax5.bar(x - width/2, ft_values, width, label='Fine-tuned')
    ax5.bar(x + width/2, orig_values, width, label='Original')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.set_ylabel('Value')
    ax5.set_title('Metrics Comparison')
    ax5.legend()
    
    # Add a text description
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    improvement_text = "\n".join([
        f"Metrics Improvement:",
        f"Chamfer Distance: {metrics_orig['chamfer_distance']:.4f} → {metrics_ft['chamfer_distance']:.4f} ({(1 - metrics_ft['chamfer_distance'] / metrics_orig['chamfer_distance']) * 100:.1f}% better)",
        f"IoU: {metrics_orig['iou']:.4f} → {metrics_ft['iou']:.4f} ({(metrics_ft['iou'] / metrics_orig['iou'] - 1) * 100:.1f}% better)",
        f"F1 Score: {metrics_orig['f1_score']:.4f} → {metrics_ft['f1_score']:.4f} ({(metrics_ft['f1_score'] / metrics_orig['f1_score'] - 1) * 100:.1f}% better)"
    ])
    ax6.text(0.0, 0.5, improvement_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{obj_id}_comparison.png"))
    plt.close()
    
    # Also save individual meshes for further inspection
    pred_mesh_ft.export(os.path.join(save_dir, f"{obj_id}_finetuned.obj"))
    pred_mesh_orig.export(os.path.join(save_dir, f"{obj_id}_original.obj"))
    target_mesh.export(os.path.join(save_dir, f"{obj_id}_ground_truth.obj"))


def evaluate():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate TripoSR fine-tuned model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to fine-tuned model")
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
    
    # Initialize models
    logger.info("Loading fine-tuned model...")
    finetuned_model = TSR(config)
    finetuned_model.load_state_dict(torch.load(args.finetuned_model, map_location=device))
    finetuned_model.to(device)
    finetuned_model.eval()
    
    logger.info("Loading original pretrained model...")
    original_model = TSR.from_pretrained(
        "TrianC0de/TripoSR",
        config_name="config.yaml",
        weight_name="sdfusion-snet-all.pth"
    )
    original_model.to(device)
    original_model.eval()
    
    # Set up data loader (using validation data)
    logger.info("Setting up data loader...")
    _, val_loader = get_dataloaders(config)
    
    # Prepare for evaluation
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            if i >= args.num_samples:
                break
                
            # Move data to device
            images = batch["images"].to(device)
            vertices_list = [v.to(device) for v in batch["vertices"]]
            faces_list = [f.to(device) for f in batch["faces"]]
            obj_ids = batch["obj_ids"]
            
            # Forward pass with fine-tuned model
            scene_codes_ft = finetuned_model(images, device=device)
            pred_meshes_ft = finetuned_model.extract_mesh(scene_codes_ft, has_vertex_color=False)
            
            # Forward pass with original model
            scene_codes_orig = original_model(images, device=device)
            pred_meshes_orig = original_model.extract_mesh(scene_codes_orig, has_vertex_color=False)
            
            # Compute metrics and visualize for each sample in batch
            for j, (pred_mesh_ft, pred_mesh_orig, target_vertices, target_faces, obj_id) in enumerate(
                zip(pred_meshes_ft, pred_meshes_orig, vertices_list, faces_list, obj_ids)
            ):
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
                
                # Visualize if requested
                if args.visualize:
                    visualize_comparison(
                        images[j],
                        pred_mesh_ft,
                        pred_mesh_orig,
                        target_vertices,
                        target_faces,
                        metrics_ft,
                        metrics_orig,
                        obj_id,
                        args.output_dir
                    )
    
    # Create a summary report
    df = pd.DataFrame(results)
    
    # Calculate average improvements
    avg_cd_improvement = df['cd_improvement'].mean()
    avg_iou_improvement = df['iou_improvement'].mean()
    avg_f1_improvement = df['f1_improvement'].mean()
    
    # Save detailed results to CSV
    df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    
    # Generate summary plots
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.boxplot([df['orig_chamfer_distance'], df['ft_chamfer_distance']])
    plt.xticks([1, 2], ['Original', 'Fine-tuned'])
    plt.title('Chamfer Distance (lower is better)')
    
    plt.subplot(1, 3, 2)
    plt.boxplot([df['orig_iou'], df['ft_iou']])
    plt.xticks([1, 2], ['Original', 'Fine-tuned'])
    plt.title('IoU (higher is better)')
    
    plt.subplot(1, 3, 3)
    plt.boxplot([df['orig_f1_score'], df['ft_f1_score']])
    plt.xticks([1, 2], ['Original', 'Fine-tuned'])
    plt.title('F1 Score (higher is better)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # Generate overall improvement plot
    plt.figure(figsize=(8, 5))
    metrics = ['Chamfer Distance', 'IoU', 'F1 Score']
    improvements = [avg_cd_improvement, avg_iou_improvement, avg_f1_improvement]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    plt.bar(metrics, improvements, color=colors)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Metric')
    plt.ylabel('Improvement (%)')
    plt.title('Average Improvement by Fine-tuned Model')
    
    for i, v in enumerate(improvements):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'overall_improvement.png'))
    plt.close()
    
    # Print summary
    summary = f"""
    Evaluation Summary:
    -------------------
    Number of samples: {len(results)}
    
    Average Metrics:
    - Original Model:
      * Chamfer Distance: {df['orig_chamfer_distance'].mean():.4f}
      * IoU: {df['orig_iou'].mean():.4f}
      * F1 Score: {df['orig_f1_score'].mean():.4f}
    
    - Fine-tuned Model:
      * Chamfer Distance: {df['ft_chamfer_distance'].mean():.4f}
      * IoU: {df['ft_iou'].mean():.4f}
      * F1 Score: {df['ft_f1_score'].mean():.4f}
    
    Average Improvements:
    - Chamfer Distance: {avg_cd_improvement:.1f}% improvement
    - IoU: {avg_iou_improvement:.1f}% improvement
    - F1 Score: {avg_f1_improvement:.1f}% improvement
    """
    
    logger.info(summary)
    
    # Save summary to file
    with open(os.path.join(args.output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(summary)


if __name__ == "__main__":
    evaluate() 