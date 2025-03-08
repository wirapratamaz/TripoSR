# 3D Model Evaluation Metrics in TripoSR

This document explains the evaluation metrics used in the TripoSR interface to assess the quality of generated 3D models.

## Comparison Metrics

These metrics evaluate the quality of reconstruction by comparing the generated model to an ideal model (when available).

### 1. Uniform Hausdorff Distance (UHD)

- **Description**: Measures the maximum distance between points in the predicted mesh and an ideal mesh. Specifically, it calculates the maximum of all minimum distances between points.
- **Range**: 0.0 to ∞ (lower is better)
- **Interpretation**: A value close to 0 indicates that even the most distant points between the surfaces are close, suggesting high shape similarity.
- **Note**: Without ground truth, this metric estimates the maximum deviation from a simplified version of the mesh.
- **Mathematical definition**: UHD = max(max(min(d(p, gt)) for p in predicted), max(min(d(gt, p)) for gt in ground_truth))

### 2. Tangent-Space Mean Distance (TMD)

- **Description**: Measures the average distance between points when projected onto the tangent plane, emphasizing local surface alignment rather than global positioning.
- **Range**: 0.0 to ∞ (lower is better)
- **Interpretation**: A value close to 0 indicates that the model's surfaces have similar local geometric properties as the ground truth.
- **Note**: This metric is more sensitive to fine geometric details than purely point-based metrics.
- **Mathematical definition**: The average of distances between points after projecting displacements to tangent spaces.

### 3. Chamfer Distance (CD)

- **Description**: Measures the average distance between points in the predicted mesh and an ideal mesh.
- **Range**: 0.0 to ∞ (lower is better)
- **Interpretation**: A value close to 0 indicates that the model's surfaces closely align with expected positions.
- **Note**: Without ground truth, this metric estimates spatial consistency of the model.

### 4. Intersection over Union (IoU)

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