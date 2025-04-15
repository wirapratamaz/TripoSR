# TripoSR Fine-Tuning Colab Notebook

## Cell 1: Setup and Dependencies

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository
!git clone https://github.com/wirapratamaz/TripoSR.git
%cd /content/TripoSR

!git checkout training
!git pull origin training

# Install dependencies
!pip install -q trimesh omegaconf einops rembg huggingface-hub==0.26.0 transformers==4.35.0
!pip install -q git+https://github.com/tatsy/torchmcubes.git
!pip install -q xatlas==0.0.9 imageio[ffmpeg] matplotlib pandas tqdm
!pip install -q moderngl scipy>=1.11.0
!pip install -r requirements.txt

# Check CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

## Cell 2: Create Directories

# Create directories
!mkdir -p /content/TripoSR/dataset/train
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository
!git clone https://github.com/wirapratamaz/TripoSR.git
%cd /content/TripoSR

# Install dependencies
!pip install -q trimesh omegaconf einops rembg huggingface-hub==0.26.0 transformers==4.35.0
!pip install -q git+https://github.com/tatsy/torchmcubes.git
!pip install -q xatlas==0.0.9 imageio[ffmpeg] matplotlib pandas tqdm
!pip install -q moderngl scipy>=1.11.0
!pip install -r requirements.txt

# Check CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
## Cell 3: Upload Config File

# Upload config.yaml rather than trying to create it inline
from google.colab import files
import io
import os

# Either upload a config file or use the default one
print("You can upload your own config.yaml or use the default one")
print("To use the default, just click 'Skip' below")

try:
  uploaded = files.upload()  # This will prompt for file upload
  if 'config.yaml' in uploaded:
    print("Using uploaded config.yaml")
  else:
    # If they uploaded something else or skipped
    raise Exception("No config.yaml uploaded, using default")
except:
  # Create the default config.yaml file using Python file operations
  # This avoids the %%writefile magic issues
  config_content = """data:
  train_path: ./dataset/train
  val_path: ./dataset/val
  input_format: image
  target_format: mesh
  resolution: 128
  num_workers: 2

training:
  batch_size: 2
  epochs: 30
  learning_rate: 1e-4
  save_interval: 5
  log_interval: 10

model:
  type: TSR
  transformer:
    encoder_layers: 12
    decoder_layers: 8
    embed_dim: 768

# Original TripoSR model configuration
cond_image_size: 256

image_tokenizer_cls: tsr.models.image_encoders.openai.OpenAIImageEncoder
image_tokenizer:
  embed_dim: 768
  encoder_pretrained: stabilityai/sd-vae-ft-ema
  encoder_config:
    z_channels: 4
    resolution: 256
    in_channels: 3
    out_ch: 3
    ch: 128
    ch_mult: [1, 2, 4, 4]
    num_res_blocks: 2
    attn_resolutions: [32]
    dropout: 0.0

tokenizer_cls: tsr.models.tokenizers.surface_plane.SurfacePlaneTokenizer
tokenizer:
  n_point_samples: 6144
  sample_mode: grid
  resolution: 32
  padding: 0.1
  embed_dim: 768
  
backbone_cls: tsr.models.transformer.Transformer
backbone:
  encoder:
    embed_dim: 768
    depth: 12
    num_heads: 12
    mlp_ratio: 4
    qkv_bias: True
  decoder:
    embed_dim: 768
    depth: 8
    num_heads: 12
    mlp_ratio: 4
    qkv_bias: True

post_processor_cls: tsr.models.post_processors.identity.Identity
post_processor: {}

decoder_cls: tsr.models.decoders.triplane.Triplane
decoder:
  dims_3d: [32, 32, 32]
  dims_2d: [256, 256]
  feat_dim: 32
  mlp_dim: 128
  out_dim: 4
  n_blocks: 2
  
renderer_cls: tsr.models.renderers.volume.VolumeRenderer
renderer:
  radius: 1.3
  n_samples: 128"""

  with open('/content/TripoSR/config.yaml', 'w') as f:
    f.write(config_content)
  
  print("Created default config.yaml")

# Verify the config file
!cat /content/TripoSR/config.yaml | head -n 10
print("... (config file continues)")

## Cell 6: Create Sample Dataset (for testing)

# Create a minimal test dataset
!mkdir -p /content/TripoSR/dataset/train/sample1
!mkdir -p /content/TripoSR/dataset/val/sample2

# Download sample images
!wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/train/sample1/image.png
!wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/val/sample2/image.png

# Create placeholder model files
!touch /content/TripoSR/dataset/train/sample1/model.obj
!touch /content/TripoSR/dataset/val/sample2/model.obj

print("Sample dataset created (for testing workflow only)")

## Cell 7: Update Model Source to Use TrianC0de/TripoSR

# Update train.py to use TrianC0de/TripoSR model
import re
with open('train.py', 'r') as f:
    train_content = f.read()

if 'TrianC0de/TripoSR' not in train_content:
    updated_content = re.sub(r'("stabilityai/TripoSR")', r'"TrianC0de/TripoSR"', train_content)
    with open('train.py', 'w') as f:
        f.write(updated_content)
    print("Updated train.py to use TrianC0de/TripoSR model")

# Do the same for evaluate.py
with open('evaluate.py', 'r') as f:
    eval_content = f.read()

if 'TrianC0de/TripoSR' not in eval_content:
    updated_content = re.sub(r'("stabilityai/TripoSR")', r'"TrianC0de/TripoSR"', eval_content)
    with open('evaluate.py', 'w') as f:
        f.write(updated_content)
    print("Updated evaluate.py to use TrianC0de/TripoSR model")

## Cell 8: Run Training

# Start fine-tuning with pretrained TrianC0de/TripoSR model
!python train.py --config config.yaml --output_dir /content/TripoSR/output --device cuda:0 --pretrained

## Cell 9: Evaluate Model

# Evaluate the fine-tuned model
!python evaluate.py --config config.yaml --finetuned_model /content/TripoSR/output/model_final.pth --output_dir /content/TripoSR/evaluation --visualize --num_samples 2

## Cell 10: Save to Google Drive

# Save the trained model to Google Drive
!cp /content/TripoSR/output/model_final.pth /content/drive/MyDrive/TripoSR_finetuned.pth
!cp -r /content/TripoSR/evaluation /content/drive/MyDrive/TripoSR_evaluation

print("Fine-tuned model and evaluation results saved to Google Drive")