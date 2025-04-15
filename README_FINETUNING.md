# Fine-Tuning TripoSR with BalineseMask3D Dataset

This repository contains the code and instructions for fine-tuning the TripoSR model using the BalineseMask3D dataset. Fine-tuning allows the model to produce higher-quality 3D reconstructions specifically tailored to cultural objects from Bali.

## Requirements

Ensure you have installed all the dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch
- trimesh
- xatlas
- torchmcubes (with CUDA support)
- omegaconf
- Pillow
- transformers
- matplotlib
- pandas
- tqdm

## Dataset Preparation

The dataset must be structured as follows:

```
dataset/
├── train/
│   ├── object_id_1/
│   │   ├── image.png     # Input RGB image
│   │   ├── model.obj     # Ground truth 3D mesh (or .ply, .off)
│   │   └── mask.png      # Optional mask
│   ├── object_id_2/
│   │   ├── ...
│   └── ...
└── val/
    ├── object_id_1/
    │   ├── image.png
    │   ├── model.obj
    │   └── mask.png
    ├── object_id_2/
    │   ├── ...
    └── ...
```

### Requirements for Meshes

- Meshes must be manifold (topologically sound with no disconnected parts)
- Meshes must be clean (free of holes or artifacts)

## Configuration

The configuration for fine-tuning is defined in `config.yaml`. Key settings to adjust:

- `data.resolution`: Resolution for input images (start with 128 for initial testing)
- `training.batch_size`: Batch size (start with 2-4 for initial testing)
- `training.epochs`: Number of epochs to train
- `training.learning_rate`: Learning rate

## Fine-Tuning Process

### Step 1: Prepare your dataset

Organize your data in the required structure as described above.

### Step 2: Train the model

Run the training script:

```bash
python train.py --config config.yaml --output_dir output --device cuda:0
```

To use transfer learning with the pretrained model:

```bash
python train.py --config config.yaml --output_dir output --device cuda:0 --pretrained
```

Arguments:
- `--config`: Path to config file
- `--output_dir`: Directory to save outputs
- `--device`: Device to use (cuda:0, cuda:1, cpu, etc.)
- `--pretrained`: Use the pretrained model for transfer learning (recommended)

### Step 3: Evaluate the model

After training, evaluate the model to compare its performance with the original pretrained model:

```bash
python evaluate.py --config config.yaml --finetuned_model output/model_final.pth --output_dir evaluation --visualize
```

Arguments:
- `--config`: Path to config file
- `--finetuned_model`: Path to the fine-tuned model checkpoint
- `--output_dir`: Directory to save evaluation results
- `--visualize`: Generate visualizations comparing original and fine-tuned model outputs
- `--num_samples`: Number of samples to evaluate (default: 10)

### Step 4: Visualize results

Generate 3D models and visualizations using your fine-tuned model:

```bash
python visualize.py --model_path output/model_final.pth --input_dir test_images --output_dir visualization
```

Arguments:
- `--model_path`: Path to the fine-tuned model checkpoint
- `--input_dir`: Directory containing input images
- `--output_dir`: Directory to save output visualizations
- `--views`: Number of views to render (default: 30)
- `--resolution`: Mesh resolution (default: 256)
- `--bake_texture`: Bake texture atlas for mesh

## Training Monitoring

During training:
- Loss curves are saved in `output/loss_plot.png`
- Sample visualizations are saved every 10 epochs
- Checkpoints are saved in `output/checkpoints/` at the interval specified in `config.yaml`

## Evaluation Metrics

The evaluation compares the fine-tuned model with the original pretrained model using:
- **Chamfer Distance**: Measures the distance between predicted and ground truth point clouds (lower is better)
- **Intersection over Union (IoU)**: Evaluates overlap between predicted and ground truth meshes (higher is better)
- **F1 Score**: Assesses accuracy on point clouds (higher is better)

Results are saved in `evaluation/evaluation_results.csv` and visualization plots in `evaluation/metrics_comparison.png`.

## Advanced Usage

### Adjusting Hyper-parameters

For better performance, you may want to experiment with:
- Learning rate schedule (already implemented as CosineAnnealingLR)
- Batch size (increase if you have more GPU memory)
- Data resolution (increase for more detail, but requires more memory)
- Number of training epochs

### Troubleshooting

Common issues:
- **Out of memory errors**: Reduce batch size or data resolution
- **Slow training**: Check if GPU acceleration is properly set up
- **Poor results**: Ensure dataset quality and try different hyperparameters

## License

This project maintains the original TripoSR license. See LICENSE file for details.

## Acknowledgments

- The original TripoSR model by StabilityAI
- The BalineseMask3D dataset creators 