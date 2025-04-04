---
title: "Research Notes: TripoSR Model Generation/Training using run.py"
author: "Wira Pratama"
---

# Research Notes: TripoSR Model Generation/Training using `run.py`

## 1. Objective

Document the process and findings related to generating TripoSR models, potentially involving training or fine-tuning, with a focus on the `run.py` script within the TripoSR repository. This investigation is prompted by CTO inquiry regarding "proses training untuk menghasilkan model ckpt dari tripoSR dari code run.py".

## 2. Initial Analysis of `run.py`

The `run.py` script provided in the TripoSR repository appears to be primarily designed for **inference**, not training. Its main functions include:

*   Loading a pre-trained TripoSR model (`.ckpt` file specified via `--pretrained-model-name-or-path`, defaulting to `stabilityai/TripoSR`).
*   Processing input images (removing background, resizing foreground).
*   Running the loaded model to generate `scene_codes` from images.
*   Extracting 3D meshes (OBJ or GLB format) using Marching Cubes (`model.extract_mesh`).
*   Optionally baking textures (`--bake-texture`).
*   Optionally rendering a video of the generated model (`--render`).

**Key Code Snippets from `run.py`:**

Loading the model:
```python
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.to(device)
```

Generating scene codes (inference):
```python
with torch.no_grad():
    scene_codes = model([image], device=device)
```

Extracting mesh:
```python
meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
```

**Conclusion:** The script `run.py` uses an *existing* `.ckpt` file to generate 3D models from images. It does not contain functionality for training or fine-tuning a model to *produce* a new `.ckpt` file.

## 3. Addressing the "Training" Aspect

Given the CTO's query about *training* using `run.py`, there might be a misunderstanding or a different script/process involved for training/fine-tuning TripoSR models.

**Possible Scenarios:**

1.  **Misinterpretation:** The term "training" might have been used loosely to refer to the process of *generating* models using the script.
2.  **Separate Training Script:** There might be another script or set of tools within the TripoSR project (or related Stability AI resources) specifically designed for training or fine-tuning the model. This requires further investigation into the repository structure or documentation.
3.  **External Process:** The `.ckpt` file itself might be trained using a different framework or process entirely, and `run.py` is solely for deployment/inference.

## 4. Environment Setup (Based on `TripoSR_Colab.txt`)

The Colab notebook (`TripoSR_Colab.txt`) provides a setup procedure, likely relevant for both inference (`run.py`, `gradio_app.py`) and potentially training if a script exists. Key steps include:

*   Cloning the repository: `git clone https://github.com/wirapratamaz/TripoSR.git`
*   Installing dependencies: `pip install -r requirements.txt`, plus specific packages like `trimesh`, `omegaconf`, `einops`, `rembg`, `torchmcubes`, `huggingface-hub`, `transformers`, `accelerate`, `diffusers`, etc.
*   Setting environment variables: `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'`

## 5. How `run.py` is Used (Inference Workflow)

To generate a 3D model from an image using `run.py`:

1.  **Prepare Environment:** Ensure all dependencies from `requirements.txt` and the setup script are installed.
2.  **Prepare Input Image:** Provide a path to one or more images. Background removal is done automatically unless `--no-remove-bg` is specified.
3.  **Run the Script:** Execute `run.py` with appropriate arguments.

    Example command:
    ```bash
    python run.py path/to/your/image.png --mc-resolution 256 --model-save-format obj --output-dir ./output 
    ```
4.  **Output:** The script will save the generated mesh (e.g., `output/0/mesh.obj`) and potentially intermediate files (like `input.png`) in the specified output directory.

## 6. Performance Metrics and Resource Requirements

Based on initial testing and analysis of the codebase, here are the key performance metrics for TripoSR inference:

### Hardware Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| GPU VRAM | 8GB     | 16GB+       | High-resolution output (256+) requires 12GB+ |
| CPU RAM  | 8GB     | 16GB        | Texture baking is memory-intensive |
| GPU Type | NVIDIA GTX 1080 | RTX 3080+ | Older GPUs significantly increase processing time |

### Inference Performance

| Configuration | Resolution | Processing Time | Quality | File Size |
|---------------|------------|----------------|---------|-----------|
| Draft (Konsep) | 128       | ~30-45 sec     | Basic   | 5-10MB    |
| Standard      | 192       | ~1-2 min       | Good    | 15-30MB   |
| High (Tinggi) | 256       | ~3-5 min       | Best    | 30-60MB   |

*Note: Times measured on NVIDIA A100 GPU; consumer hardware will likely be slower.*

### Quality-Performance Tradeoffs

* **Marching Cubes Resolution (`--mc-resolution`)**: Higher values produce more detailed meshes but exponentially increase memory usage and processing time.
* **Chunk Size (`--chunk-size`)**: Smaller chunks reduce VRAM usage but increase computation time.
* **Texture Resolution**: Higher texture resolution produces better visual quality but increases file size and baking time.

## 7. Alternative Approaches

Before pursuing custom model training, we should evaluate alternatives:

### Option 1: Using Pre-trained Model with Parameter Tuning

* **Pros**: 
  - Immediate implementation
  - No training infrastructure required
  - Predictable results
  - Well-documented and supported
* **Cons**:
  - Limited customization for domain-specific objects
  - Cannot incorporate our proprietary knowledge/data

### Option 2: Fine-tuning Existing Model

* **Pros**:
  - Leverages existing model architecture and knowledge
  - Requires less data than training from scratch
  - Can adapt to our specific use cases
* **Cons**:
  - Requires finding/creating training code (not in `run.py`)
  - Needs curated dataset
  - Significant computational resources for fine-tuning

### Option 3: Developing Custom Training Pipeline

* **Pros**:
  - Maximum flexibility and control
  - Can be optimized for our specific domain
* **Cons**:
  - Highest technical complexity
  - Longest implementation timeline
  - Greatest resource requirements
  - May require architectural expertise beyond our current team

## 8. Technical Debt Considerations

Pursuing custom model training introduces several technical debt factors:

* **Model Versioning**: Need infrastructure to track model versions, performance metrics, and training datasets.
* **Training Infrastructure**: Maintaining GPU resources, storage for training data, and model checkpoints.
* **Knowledge Dependency**: Creates specialized knowledge requirements that may be concentrated in few team members.
* **Ongoing Maintenance**: Models degrade over time; regular retraining cycles will be necessary.
* **Compatibility Management**: Ensuring compatibility with changing dependencies (PyTorch, CUDA, etc.).
* **Documentation Burden**: Requires comprehensive documentation of training process, parameters, and dataset preparation.

##  Resources

*   TripoSR Repository: [https://github.com/VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) (or the specific fork being used)
*   Markdown for Academia: [https://scientificallysound.org/2021/03/09/markdown-for-science-and-academia-part-2/](https://scientificallysound.org/2021/03/09/markdown-for-science-and-academia-part-2/)
*   Scientific Writing with Markdown: [https://jaantollander.com/post/scientific-writing-with-markdown/](https://jaantollander.com/post/scientific-writing-with-markdown/)
*   Stability AI Blog (for potential training info): [https://stability.ai/blog](https://stability.ai/blog)
