import argparse
import numpy as np
import trimesh
import logging
from tsr.evaluation import (
    calculate_metrics, 
    calculate_f1_score, 
    calculate_chamfer_distance, 
    calculate_iou, 
    calculate_optimal_sample_points,
    calculate_tangent_space_mean_distance
)
import sys
import time

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

def test_adaptive_sampling():
    """Test the adaptive sampling functionality."""
    logging.info("Testing adaptive sampling with meshes of different complexities...")
    
    # Create meshes of different complexities
    print("\n===== Adaptive Sampling Tests =====")
    
    # Low complexity - simple box
    low_mesh = trimesh.creation.box(extents=[1, 1, 1])
    low_sample_count = calculate_optimal_sample_points(low_mesh)
    print(f"Low complexity mesh: {len(low_mesh.vertices)} vertices, {len(low_mesh.faces)} faces")
    print(f"Optimal sample count: {low_sample_count}")
    
    # Medium complexity - icosphere
    medium_mesh = trimesh.creation.icosphere(subdivisions=2)
    medium_sample_count = calculate_optimal_sample_points(medium_mesh)
    print(f"Medium complexity mesh: {len(medium_mesh.vertices)} vertices, {len(medium_mesh.faces)} faces")
    print(f"Optimal sample count: {medium_sample_count}")
    
    # High complexity - icosphere with more subdivisions
    high_mesh = trimesh.creation.icosphere(subdivisions=4)
    high_sample_count = calculate_optimal_sample_points(high_mesh)
    print(f"High complexity mesh: {len(high_mesh.vertices)} vertices, {len(high_mesh.faces)} faces")
    print(f"Optimal sample count: {high_sample_count}")
    
    # Verify that more complex meshes get more sample points
    assert low_sample_count <= medium_sample_count <= high_sample_count, \
        "Sample counts should increase with mesh complexity"
    
    print("Adaptive sampling test passed!")
    
    return low_mesh, medium_mesh, high_mesh

def test_tmd_performance():
    """Test the performance of the enhanced TMD calculation."""
    logging.info("Testing TMD calculation performance...")
    
    print("\n===== TMD Performance Tests =====")
    
    # Create test meshes
    # Use a complex shape and a slightly perturbed version of it
    source_mesh = trimesh.creation.icosphere(subdivisions=3)
    
    # Create a perturbed version by adding noise to vertices
    noise = np.random.normal(0, 0.02, source_mesh.vertices.shape)
    target_mesh = source_mesh.copy()
    target_mesh.vertices += noise
    
    # Measure time for TMD calculation
    start_time = time.time()
    tmd = calculate_tangent_space_mean_distance(source_mesh, target_mesh)
    end_time = time.time()
    
    # Print results
    print(f"TMD between original and perturbed mesh: {tmd:.6f}")
    print(f"Calculation time: {(end_time - start_time):.4f} seconds")
    
    return tmd

def main():
    parser = argparse.ArgumentParser(description="Test the 3D model evaluation metrics.")
    parser.add_argument("--model", type=str, help="Path to the model to evaluate")
    parser.add_argument("--reference", type=str, help="Path to the reference model (optional)", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run tests with synthetic data")
    parser.add_argument("--adaptive", action="store_true", help="Test adaptive sampling and TMD enhancements")
    
    args = parser.parse_args()
    
    if args.synthetic:
        test_individual_metrics()
    
    if args.adaptive:
        test_meshes = test_adaptive_sampling()
        tmd_result = test_tmd_performance()
    
    if args.model:
        test_metrics(args.model, args.reference)
    
    if not args.model and not args.synthetic and not args.adaptive:
        logging.error("Please provide either a model path with --model, use --synthetic for synthetic tests, or use --adaptive for adaptive sampling tests")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 