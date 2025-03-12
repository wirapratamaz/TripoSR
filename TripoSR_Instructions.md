# TripoSR Google Colab Instructions

## Setup Instructions

Copy and paste the following code blocks into separate cells in a Google Colab notebook:

### Cell 1: Setup and Installation

```python
# Install required dependencies
%cd /content

# Clone the repository from your fork
!git clone https://github.com/wirapratamaz/TripoSR.git
%cd /content/TripoSR

# Install basic dependencies
!pip install -q trimesh omegaconf einops rembg gradio

# Install torchmcubes with CUDA support
!pip install -q git+https://github.com/tatsy/torchmcubes.git

# Install additional dependencies from requirements.txt
!pip install -r requirements.txt

# Install specific versions to ensure compatibility
!pip install huggingface-hub==0.26.0
!pip install transformers==4.35.0
!pip install accelerate==0.20.3

# Install additional dependencies
!pip install -q xatlas==0.0.9
!pip install -q imageio[ffmpeg]
!pip install -q onnxruntime

# Ensure we have the latest gradio
!pip install --upgrade gradio

# Set up environment variables for better GPU memory management
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Cell 2: Run the Gradio App

```python
# Run the Gradio app with lower resolution for faster processing
%cd /content/TripoSR
!python main.py --queuesize 1 --share
```

## Troubleshooting Blank Model Output

If you're experiencing blank model output after 30 minutes, try these solutions:

1. **Ensure you're using a GPU runtime**
   - Go to Runtime > Change runtime type > Hardware accelerator > GPU

2. **Lower the Marching Cubes Resolution**
   - In the Gradio interface, set the "Marching Cubes Resolution" slider to 128 or 64
   - This will produce a lower quality model but should complete faster

3. **Modify the chunk size**
   - Add this line to Cell 2 before running the app:
   ```python
   !python main.py --queuesize 1 --share --chunk-size 4096
   ```

4. **Check your downloaded OBJ file**
   - Even if the model appears blank in the interface, the downloaded .obj file might be valid
   - Try opening it with a 3D viewer application like Blender or online viewers

5. **Try with example images first**
   - Use one of the example images provided in the interface before trying your own

6. **Restart the runtime if needed**
   - If the process seems stuck, go to Runtime > Restart runtime and try again

## Alternative Direct Run Method

If the Gradio interface is too slow, you can process images directly:

```python
%cd /content/TripoSR
!python run.py examples/tiger_girl.png --mc-resolution 128 --output-dir output/
```

This will process the image and save the result to the output directory, which you can then download.
