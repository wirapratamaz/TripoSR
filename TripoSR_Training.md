# TripoSR Fine-Tuning using BalineseMask3D Dataset

# 1. Setup and dependencies
%%capture
# Mount Google Drive for storing datasets and checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository (choose the appropriate branch)
!git clone https://github.com/wirapratamaz/TripoSR.git
%cd /content/TripoSR

# Use the training branch if it exists, otherwise use main
!git checkout training

# Install all required dependencies
!pip install -q trimesh omegaconf einops rembg
!pip install -q git+https://github.com/tatsy/torchmcubes.git
!pip install huggingface-hub==0.26.0
!pip install transformers==4.35.0
!pip install accelerate==0.20.3
!pip install diffusers==0.14.0
!pip install -q xatlas==0.0.9
!pip install -q imageio[ffmpeg]
!pip install -q onnxruntime
!pip install scipy>=1.11.0
!pip install matplotlib pandas tqdm
!pip install -q aiofiles fastapi orjson typing-extensions
!pip install -q moderngl
!pip install -r requirements.txt

# Set up environment variables for better GPU memory management
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Verify CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 2. Create the necessary directories for data and output
!mkdir -p /content/TripoSR/dataset/train
!mkdir -p /content/TripoSR/dataset/val
!mkdir -p /content/TripoSR/output
!mkdir -p /content/TripoSR/evaluation

# 3. Create/modify the config.yaml file
%%writefile /content/TripoSR/config.yaml
data:
  train_path: ./dataset/train
  val_path: ./dataset/val
  input_format: image
  target_format: mesh
  resolution: 128
  num_workers: 2  # Lower for Colab

training:
  batch_size: 2        # Adjust based on your Colab GPU
  epochs: 50           # Reduced for Colab runtime constraints
  learning_rate: 1e-4
  save_interval: 5     # Save more frequently in Colab
  log_interval: 10     # Log more frequently

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
  n_samples: 128

# 4. Dataset preparation
# Option 1: Upload dataset (if you have it prepared locally)
from google.colab import files

# Create a dataset upload helper
def upload_dataset_files():
    print("Please upload your dataset files (.zip recommended):")
    uploaded = files.upload()
    return list(uploaded.keys())[0]

# Option for dataset upload
import ipywidgets as widgets
from IPython.display import display

dataset_option = widgets.RadioButtons(
    options=['Upload zip dataset', 'Use sample data for testing'],
    description='Dataset:',
    disabled=False
)
display(dataset_option)

# Process dataset based on selection
if dataset_option.value == 'Upload zip dataset':
    zip_file = upload_dataset_files()
    if zip_file.endswith('.zip'):
        !unzip "{zip_file}" -d /content/dataset_extracted
        
        # Now run the data preparation script
        !python prepare_data.py --input_dir /content/dataset_extracted --output_dir /content/TripoSR/dataset --val_split 0.2 --resize 128
    else:
        print("Please upload a zip file containing your dataset.")
else:
    # Create a minimal test dataset for demonstration
    !mkdir -p /content/TripoSR/dataset/train/sample1
    !mkdir -p /content/TripoSR/dataset/val/sample2
    
    # Download a sample image and model for testing
    !wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/train/sample1/image.png
    !wget -q https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/assets/teapot.png -O /content/TripoSR/dataset/val/sample2/image.png
    
    # For the 3D models, we'll just create placeholder files since we don't have actual model files
    !touch /content/TripoSR/dataset/train/sample1/model.obj
    !touch /content/TripoSR/dataset/val/sample2/model.obj
    
    print("Created sample dataset for testing purposes only.")
    print("⚠️ Note: This won't produce meaningful results - just for testing the workflow!")

# 5. Test the dataset loading
!python test_data.py --config config.yaml --output_dir dataset_test --num_samples 2

# 6. Fine-tune the model using the TrianC0de/TripoSR checkpoint
# Make sure the train.py has been modified to use the TrianC0de/TripoSR model

# Checking if train.py uses the correct model source
import re

with open('train.py', 'r') as f:
    train_content = f.read()

if 'TrianC0de/TripoSR' not in train_content:
    # Update the train.py file to use the TrianC0de/TripoSR model
    updated_content = re.sub(r'("stabilityai/TripoSR")', r'"TrianC0de/TripoSR"', train_content)
    
    with open('train.py', 'w') as f:
        f.write(updated_content)
    
    print("Updated train.py to use TrianC0de/TripoSR model")

# Start the fine-tuning process with reduced epochs for Colab
!python train.py --config config.yaml --output_dir /content/TripoSR/output --device cuda:0 --pretrained

# 7. Evaluate the fine-tuned model
!python evaluate.py --config config.yaml --finetuned_model /content/TripoSR/output/model_final.pth --output_dir /content/TripoSR/evaluation --visualize --num_samples 5

# 8. Visualize some results with the fine-tuned model
# First upload some test images
from google.colab import files

def upload_test_images():
    print("Please upload test images:")
    uploaded = files.upload()
    
    # Save uploaded images to a directory
    import os
    test_dir = '/content/TripoSR/test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    for filename in uploaded.keys():
        with open(os.path.join(test_dir, filename), 'wb') as f:
            f.write(uploaded[filename])
    
    return test_dir

test_dir = upload_test_images()

# Run visualization on the uploaded images
!python visualize.py --model_path /content/TripoSR/output/model_final.pth --input_dir {test_dir} --output_dir /content/TripoSR/visualization

# 9. Save the fine-tuned model to Google Drive
!cp /content/TripoSR/output/model_final.pth /content/drive/MyDrive/TripoSR_finetuned.pth
!cp -r /content/TripoSR/evaluation /content/drive/MyDrive/TripoSR_evaluation
!cp -r /content/TripoSR/visualization /content/drive/MyDrive/TripoSR_visualization

print("Fine-tuned model and results saved to Google Drive")

# 10. Download the results (optional)
from google.colab import files

# Zip the results for easy download
!zip -r /content/TripoSR_results.zip /content/TripoSR/output /content/TripoSR/evaluation /content/TripoSR/visualization

# Initiate download
files.download('/content/TripoSR_results.zip')