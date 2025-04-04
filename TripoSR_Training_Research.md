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

## 6. Resources

*   TripoSR Repository: [https://github.com/VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) (or the specific fork being used)
*   Markdown for Academia: [https://scientificallysound.org/2021/03/09/markdown-for-science-and-academia-part-2/](https://scientificallysound.org/2021/03/09/markdown-for-science-and-academia-part-2/)
*   Scientific Writing with Markdown: [https://jaantollander.com/post/scientific-writing-with-markdown/](https://jaantollander.com/post/scientific-writing-with-markdown/)
