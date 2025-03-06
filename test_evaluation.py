import argparse
import numpy as np
import trimesh
import logging
from tsr.evaluation import calculate_metrics, calculate_f1_score, calculate_chamfer_distance, calculate_iou
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_metrics(model_path, reference_path=None):
    """Test the evaluation metrics on a given model file."""
    logging.info(f"Loading model: {model_path}")
    
    try:
        # Load the model
        mesh = trimesh.load(model_path)
        logging.info(f"Loaded model with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Load reference model if provided
        ground_truth_mesh = None
        if reference_path:
            try:
                ground_truth_mesh = trimesh.load(reference_path)
                logging.info(f"Loaded reference model with {len(ground_truth_mesh.vertices)} vertices and {len(ground_truth_mesh.faces)} faces")
            except Exception as e:
                logging.error(f"Error loading reference model: {str(e)}")
                ground_truth_mesh = None
        
        # Calculate metrics
        logging.info("Calculating metrics...")
        metrics = calculate_metrics(mesh, ground_truth_mesh)
        
        # Print metrics
        print("\n===== Evaluation Metrics =====")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Chamfer Distance: {metrics['chamfer_distance']:.4f}")
        print(f"IoU Score: {metrics['iou_score']:.4f}")
        
        print("\n===== Mesh Quality =====")
        print(f"Vertices: {metrics['vertices']}")
        print(f"Faces: {metrics['faces']}")
        print(f"Watertight: {'Yes' if metrics['watertight'] > 0.5 else 'No'}")
        print(f"Manifold: {'Yes' if metrics['manifold'] > 0.5 else 'No'}")
        print(f"Regularity: {metrics['regularity']:.4f}")
        print(f"Area Uniformity: {metrics['area_uniformity']:.4f}")
        
        if ground_truth_mesh is None:
            print("\nNote: Metrics are estimates as no reference model was provided.")
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        return None

def test_individual_metrics():
    """Test the individual metric functions with synthetic data."""
    logging.info("Testing individual metrics with synthetic data...")
    
    # Create simple test data
    pred_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    
    gt_points = np.array([
        [0.1, 0.1, 0.1],
        [1.1, 0.1, 0.1],
        [0.1, 1.1, 0.1],
        [1.1, 1.1, 0.1]
    ])
    
    # Test F1 score
    f1 = calculate_f1_score(pred_points, gt_points, threshold=0.2)
    print(f"\nF1 Score (synthetic): {f1:.4f}")
    
    # Test Chamfer distance
    cd = calculate_chamfer_distance(pred_points, gt_points)
    print(f"Chamfer Distance (synthetic): {cd:.4f}")
    
    # Create simple meshes for IoU test
    pred_mesh = trimesh.creation.box(extents=[1, 1, 1])
    gt_mesh = trimesh.creation.box(extents=[1.2, 1.2, 1.2])
    gt_mesh.apply_translation([0.1, 0.1, 0.1])  # Slight offset
    
    # Test IoU
    iou = calculate_iou(pred_mesh, gt_mesh)
    print(f"IoU Score (synthetic): {iou:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Test the 3D model evaluation metrics.")
    parser.add_argument("--model", type=str, help="Path to the model to evaluate")
    parser.add_argument("--reference", type=str, help="Path to the reference model (optional)", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run tests with synthetic data")
    
    args = parser.parse_args()
    
    if args.synthetic:
        test_individual_metrics()
    
    if args.model:
        test_metrics(args.model, args.reference)
    
    if not args.model and not args.synthetic:
        logging.error("Please provide either a model path with --model or use --synthetic for synthetic tests")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 