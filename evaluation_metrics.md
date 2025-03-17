# 3D Model Evaluation Metrics in TripoSR

This document explains the evaluation metrics used in the TripoSR interface to assess the quality of generated 3D models.

## Comparison Metrics

These metrics evaluate the quality of reconstruction by comparing the generated model to an ideal model (when available).

### 1. F1-Score

- **Description**: Combines precision and recall to measure how accurately the model's surface points match those of an ideal model.
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: A score close to 1.0 indicates that the model's surface points are well-positioned.
- **Note**: In the absence of a ground truth model, this score provides an estimate based on statistical analysis of the model's consistency.

### 2. Uniform Hausdorff Distance (UHD)

- **Description**: Measures the maximum distance between points in the predicted mesh and an ideal mesh. Specifically, it calculates the maximum of all minimum distances between points.
- **Range**: 0.0 to ∞ (lower is better)
- **Interpretation**: A value close to 0 indicates that even the most distant points between the surfaces are close, suggesting high shape similarity.
- **Note**: Without ground truth, this metric estimates the maximum deviation from a simplified version of the mesh.
- **Mathematical definition**: UHD = max(max(min(d(p, gt)) for p in predicted), max(min(d(gt, p)) for gt in ground_truth))

### 3. Tangent-Space Mean Distance (TMD)

The Tangent-Space Mean Distance measures the dissimilarity between two meshes in terms of their local surface properties. Unlike Chamfer Distance which only considers point positions, TMD considers the tangential components of the displacement vectors between mesh points, providing a more accurate measure of surface discrepancy.

### Recent Enhancements

The TMD calculation has been improved with adaptive sampling techniques to provide more accurate results for meshes of varying complexity:

1. **Dynamic Sample Point Calculation**:
   - Previously: Fixed 2000 sample points regardless of mesh complexity
   - Now: Sample count is dynamically calculated based on:
     - Surface area (larger surfaces get more samples)
     - Vertex and face count (more complex meshes get more samples)
     - Local curvature (areas with high curvature get more detailed sampling)

2. **Performance Optimizations**:
   - KD-tree based nearest neighbor search, drastically improving speed for complex meshes
   - Spatial locality exploitation using adaptive search radius
   - Fallback implementation for edge cases

3. **Robustness Improvements**:
   - Enhanced error handling for edge cases
   - Protection against numerical instabilities
   - Proper normalization of vectors

This improved implementation provides more reliable TMD measurements across meshes of different sizes and complexities, while maintaining reasonable computational performance.

### Technical Details

The sample count is calculated using:
```python
def calculate_optimal_sample_points(mesh):
    # Base count from surface area
    base_samples = int(np.sqrt(mesh.area) * 100)
    
    # Adjust for complexity
    complexity_factor = np.log10(max(1, vertex_count * face_count)) / 5
    
    # Adjust for curvature
    # Higher curvature = more samples
    curvature_factor = 1.0 + min(3.0, avg_curvature * 5.0)
    
    # Calculate and clamp final count
    sample_count = int(base_samples * complexity_factor * curvature_factor)
    return max(min_samples, min(sample_count, max_samples))
```

To test the enhanced TMD calculation, run:
```
python test_evaluation.py --adaptive
```

### 4. Chamfer Distance (CD)

- **Description**: Measures the average distance between points in the predicted mesh and an ideal mesh.
- **Range**: 0.0 to ∞ (lower is better)
- **Interpretation**: A value close to 0 indicates that the model's surfaces closely align with expected positions.
- **Note**: Without ground truth, this metric estimates spatial consistency of the model.

### 5. Intersection over Union (IoU)

- **Description**: Measures volumetric overlap between the predicted 3D model and an ideal model.
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: A score close to 1.0 indicates good volume representation compared to the expected result.
- **Note**: In the current implementation, this is an approximation based on volume comparison.

## Mesh Quality Metrics

These metrics evaluate inherent properties of the generated mesh, independent of any reference model.

### 1. Vertex and Face Count

- **Description**: Basic statistics about mesh complexity.
- **Interpretation**: Higher counts indicate more detailed meshes, but may also reflect unnecessary complexity.

### 2. Watertight

- **Description**: Indicates whether the mesh forms a closed surface without holes.
- **Interpretation**: "Yes" indicates a properly sealed mesh suitable for 3D printing or physical simulation.

### 3. Regularity

- **Description**: Measures how evenly distributed the mesh vertices and faces are.
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Higher values indicate a more uniform, clean mesh.

### 4. Area Uniformity

- **Description**: Measures consistency in the size of faces across the mesh.
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Higher values indicate more uniform face sizes throughout the mesh.

## Technical Notes

- In the absence of ground truth models for comparison, the comparison metrics provide estimations based on statistical analysis of the generated mesh.
- These metrics are computationally efficient and designed to run quickly alongside the generation process.
- For research purposes requiring precise comparison to ground truth models, users can implement custom evaluation pipelines using the exported mesh files.

## Future Enhancements

Planned improvements to the evaluation system include:

1. Support for user-provided reference models for direct comparison
2. Visualization of mesh quality through heatmaps
3. Extended metrics for texture and material quality evaluation
4. Batch evaluation support for comparing multiple generation settings 