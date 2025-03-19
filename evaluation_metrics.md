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

### 3. Total Mutual Difference (TMD)

- **Description**: Measures the diversity between different possible completions of a 3D shape by calculating the average Chamfer distance between pairs of shapes.
- **Range**: 0.0 to ∞ (higher is better)
- **Interpretation**: Higher values indicate greater diversity among generated shapes, representing a model's ability to produce varied outputs for the same input.
- **Note**: Unlike other metrics where lower values are better, for TMD, higher values are desirable as they indicate more diverse outputs.
- **Key Distinction**: TMD evaluates diversity among multiple generated shapes, not similarity to a reference shape, making it fundamentally different from metrics like CD and UHD.

#### Mathematical Definition

TMD is calculated as the average Chamfer distance between all pairs of completed shapes:

```
TMD = (1/n(n-1)) * ∑_{i=1}^{n} ∑_{j=i+1}^{n} CD(S_i, S_j)
```

Where:
- n is the number of shape completions being compared
- S_i and S_j are different completed shapes
- CD is the Chamfer distance function

#### Application in 3D Generation

TMD is particularly valuable for evaluating:
- Multi-modal shape completion tasks
- Generative models that should produce diverse outputs
- Models that capture the full spectrum of possible shapes for a given input

When using TMD for evaluation, it's important to generate multiple completions from the same input to properly measure the diversity of the model's outputs.

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