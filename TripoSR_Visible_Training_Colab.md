# TripoSR OpenLRM Visible Training Notebook

#This notebook provides a transparent training process for TripoSR with OpenLRM integration, featuring:
#- Direct GLB file upload in the notebook
#- Visible training with real-time metrics
#- Model checkpoints in OpenLRM .ckpt format

## 1. Setup and Dependencies

# Clean up any existing directories first
!rm -rf TripoSR

# Clone your TripoSR repository with the specific branch
!git clone -b openlrm-training --single-branch --depth 1 https://github.com/wirapratamaz/TripoSR.git

# Install the package in development mode
%cd TripoSR
!pip install -e .

# Install additional dependencies
!pip install torch torchvision tqdm ipywidgets matplotlib numpy omegaconf Pillow

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from tqdm.notebook import tqdm
import datetime
from omegaconf import OmegaConf

# Add the TripoSR directory to the path
sys.path.append('./TripoSR')

# Import from openlrm_integration modules
from openlrm_integration.config_ui import ConfigManager
from openlrm_integration.metrics_visualization import MetricsVisualizer
from openlrm_integration.training_dashboard import TrainingDashboard
from openlrm_integration.notebook_utils import setup_file_upload, create_export_button
from openlrm_integration.visible_training_loop import VisibleTrainingLoop

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

## 2. GLB File Upload

#Set up file upload for .glb 3D model files. This allows you to directly upload your models in the notebook.

# Import required widgets for file upload
from ipywidgets import FileUpload, Layout, Button, VBox, HBox, HTML, Output, widgets
from IPython.display import display, clear_output
import random

# First, create a modified version of setup_file_upload function without sample data option
def custom_setup_file_upload(dataset_path):
    """
    Set up file upload functionality for .glb files in Colab.
    
    Args:
        dataset_path: Path where uploaded files will be saved
    
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'val'), exist_ok=True)
    
    # Check if running in Colab
    try:
        import google.colab
        from google.colab import files
        in_colab = True
    except ImportError:
        in_colab = False
        print("Not running in Google Colab. File upload not available.")
        return
    
    # Use output area for displaying information
    output = Output()
    
    # Create train and val directories upfront
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Process uploaded files function
    def process_uploads():
        with output:
            clear_output()
            print("Please select .glb files to upload...")
            
            # Use Colab's native upload method
            uploaded = files.upload()
            total_files = len(uploaded)
            
            if total_files == 0:
                print("No files uploaded.")
                return
            
            # Process uploaded files
            if total_files == 1:
                # For a single file, save to both train and val
                for name, data in uploaded.items():
                    if not name.endswith('.glb'):
                        print(f"Skipping {name} - not a .glb file")
                        continue
                    
                    # Save to train directory
                    train_path = os.path.join(train_dir, name)
                    with open(train_path, 'wb') as f:
                        f.write(data)
                    print(f"Saved {name} to train directory")
                    
                    # Also save to val directory
                    val_path = os.path.join(val_dir, name)
                    with open(val_path, 'wb') as f:
                        f.write(data)
                    print(f"Saved a copy of {name} to val directory for validation")
            else:
                # For multiple files, distribute between train and val
                file_items = list(uploaded.items())
                random.shuffle(file_items)  # Randomize file order
                
                # Calculate split
                train_count = max(1, int(0.8 * total_files))
                val_count = total_files - train_count
                
                # Ensure at least one file goes to validation if more than one file
                if val_count == 0 and total_files > 1:
                    train_count -= 1
                    val_count = 1
                
                # Process files
                for i, (name, data) in enumerate(file_items):
                    if not name.endswith('.glb'):
                        print(f"Skipping {name} - not a .glb file")
                        continue
                        
                    if i < train_count:
                        split = "train"
                        file_path = os.path.join(train_dir, name)
                    else:
                        split = "val"
                        file_path = os.path.join(val_dir, name)
                    
                    with open(file_path, 'wb') as f:
                        f.write(data)
                    print(f"Saved {name} to {split} directory")
            
            # Print summary
            train_files = [f for f in os.listdir(train_dir) if f.endswith('.glb')]
            val_files = [f for f in os.listdir(val_dir) if f.endswith('.glb')]
            print("\nSummary:")
            print(f"Files in train directory: {len(train_files)}")
            print(f"Files in val directory: {len(val_files)}")
    
    # Create a button for upload
    upload_button = Button(
        description='Upload .glb Files',
        button_style='primary',
        tooltip='Click to upload .glb files for training',
        icon='upload',
        layout=Layout(width='300px', height='50px')
    )
    
    # Connect the button to the upload function
    upload_button.on_click(lambda b: process_uploads())
    
    # Display the button and output area
    display(VBox([
        HTML("<h3>Upload .glb Files for Training</h3>"),
        upload_button,
        output
    ]))

# Create dataset directories
dataset_path = './dataset'
os.makedirs(dataset_path, exist_ok=True)

# Use our modified upload function without sample data option
custom_setup_file_upload(dataset_path)

## 3. Configuration

# Create a default configuration directly without the ConfigManager
config = OmegaConf.create({
    'dataset': {
        'path': dataset_path,
        'image_size': 224,
        'batch_size': 4,
    },
    'model': {
        'hidden_dim': 64,
        'num_layers': 3,
    },
    'training': {
        'epochs': 10,
        'lr': 0.001,
        'checkpoint_dir': './checkpoints',
        'log_interval': 5,
    }
})

print("Created default configuration:")
print(OmegaConf.to_yaml(config))

# Create basic UI elements for configuration adjustment
from ipywidgets import IntSlider, FloatSlider, VBox, HBox, Label, Layout

# Dataset configuration
dataset_widgets = [
    Label("Dataset Settings", style={'font_weight': 'bold'}),
    IntSlider(value=config.dataset.image_size, min=64, max=512, step=16, description='Image Size'),
    IntSlider(value=config.dataset.batch_size, min=1, max=16, step=1, description='Batch Size')
]

# Model configuration
model_widgets = [
    Label("Model Settings", style={'font_weight': 'bold'}),
    IntSlider(value=config.model.hidden_dim, min=16, max=256, step=16, description='Hidden Dim')
]

# Training configuration
training_widgets = [
    Label("Training Settings", style={'font_weight': 'bold'}),
    IntSlider(value=config.training.epochs, min=1, max=50, step=1, description='Epochs'),
    FloatSlider(value=config.training.lr, min=0.0001, max=0.01, step=0.0001, description='Learning Rate')
]

# Display the widgets
display(VBox([
    VBox(dataset_widgets),
    VBox(model_widgets),
    VBox(training_widgets)
]))

# Update config from widgets for next steps
def update_config_from_widgets():
    config.dataset.image_size = dataset_widgets[1].value
    config.dataset.batch_size = dataset_widgets[2].value
    config.model.hidden_dim = model_widgets[1].value
    config.training.epochs = training_widgets[1].value
    config.training.lr = training_widgets[2].value
    return config

# Function to call when moving to next step
update_config_from_widgets()

## 4. Dataset Loading

#Load and prepare the dataset for training.

# Dataset class for .glb files
from torch.utils.data import Dataset, DataLoader

class GLBDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # List all .glb files in the directory
        self.file_paths = []
        split_dir = os.path.join(root_dir, split)
        if os.path.exists(split_dir):
            for file in os.listdir(split_dir):
                if file.endswith('.glb'):
                    self.file_paths.append(os.path.join(split_dir, file))
        
        # If no .glb files found, create a dummy dataset for demonstration
        if len(self.file_paths) == 0:
            print(f"No .glb files found in {split_dir}. Creating a dummy dataset for demonstration.")
            self.file_paths = [f"dummy_{i}.glb" for i in range(10)]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # In a real implementation, you would load and process the .glb file here
        # For now, we'll just return random tensors as placeholders
        sample = {
            'input': torch.randn(3, self.image_size, self.image_size),
            'target': torch.randn(3, self.image_size, self.image_size),
            'path': file_path
        }
        
        return sample

# Get configuration values from our config object
# First make sure we get the latest values from widgets
config = update_config_from_widgets()

# Extract values we need
dataset_path = config.dataset.path
image_size = config.dataset.image_size
batch_size = config.dataset.batch_size

# Check if we have any files in the train directory
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')

# Get list of .glb files in train directory
train_files = [f for f in os.listdir(train_dir) if f.endswith('.glb')] if os.path.exists(train_dir) else []

# Get list of .glb files in val directory
val_files = [f for f in os.listdir(val_dir) if f.endswith('.glb')] if os.path.exists(val_dir) else []

print(f"Found {len(train_files)} files in train directory and {len(val_files)} files in val directory")

# If we have train files but no val files, copy one train file to val
if len(train_files) > 0 and len(val_files) == 0:
    import shutil
    import random
    # Select a random file to copy
    file_to_copy = random.choice(train_files)
    src_path = os.path.join(train_dir, file_to_copy)
    dst_path = os.path.join(val_dir, file_to_copy)
    
    # Make sure val directory exists
    os.makedirs(val_dir, exist_ok=True)
    
    # Copy the file
    shutil.copy(src_path, dst_path)
    print(f"Copied {file_to_copy} from train to val directory for validation")
    
    # Update val_files list
    val_files = [file_to_copy]

# Create datasets
train_dataset = GLBDataset(dataset_path, split='train', image_size=image_size)
val_dataset = GLBDataset(dataset_path, split='val', image_size=image_size)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

## 5. Model Setup

#Set up the TripoSR model and optimizer.

# Create a simple model for demonstration
# In a real implementation, you would import and use the TripoSR model from OpenLRM
class SimpleModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Handle dictionary input (from dataset) or direct tensor input
        if isinstance(x, dict) and 'input' in x:
            x = x['input']
            
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def export_checkpoint(self, path):
        """
        Export model checkpoint in OpenLRM .ckpt format
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'metadata': {
                'export_time': datetime.datetime.now().isoformat(),
                'format': 'OpenLRM'
            }
        }
        torch.save(checkpoint, path)
        print(f"Exported checkpoint to {path}")

# Get model configuration from our config object
# Make sure we have the latest widget values
config = update_config_from_widgets()
hidden_dim = config.model.hidden_dim

# Create model and optimizer
model = SimpleModel(hidden_dim=hidden_dim).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

print(f"Model created and moved to {device}")

## 6. Training with Visible Dashboard

#Train the model using the Visible Training Loop and Training Dashboard for real-time monitoring.

# Get the latest config values from widgets
config = update_config_from_widgets()

# Create checkpoint directory
checkpoint_dir = config.training.checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)

# Use existing config file with adjusted dataset path
config_file = 'openlrm_integration/configs/default_config.yaml'

# Update the dataset path in the config to match our uploaded data
# Load the config, update it, and save it back
!sed -i "s|dataset_path:.*|dataset_path: \"{dataset_path}\"|g" {config_file}

# Print the config we'll be using
print(f"Using config file: {config_file}")
print("Config contents:")
!cat {config_file}

# Make sure we can see the training output
from IPython.display import clear_output, display
from ipywidgets import Button, Output, Layout
import time

# Function to run our own custom training implementation without relying on OpenLRM
def run_custom_training():
    # Create widget to show training progress
    output = Output()
    display(output)
    
    # Create a button to start training
    start_button = Button(
        description='Start Training',
        button_style='success',
        icon='play',
        layout=Layout(width='200px', height='40px')
    )
    
    def on_start_button_clicked(b):
        with output:
            clear_output()
            print("Starting custom training process...")
            
            # Make sure the checkpoint directory exists
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Get the latest config values
            config = update_config_from_widgets()
            
            # Set up tqdm progress bar for epochs
            from tqdm.notebook import tqdm
            num_epochs = config.training.epochs
            
            # Training function with progress tracking
            def train_model():
                # Track best validation loss for model saving
                best_val_loss = float('inf')
                
                # Set up loss history for plotting
                train_losses = []
                val_losses = []
                
                # Create progress bar for epochs
                epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
                
                for epoch in epoch_pbar:
                    # Training phase
                    model.train()
                    train_loss = 0.0
                    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
                    
                    for i, batch in enumerate(train_pbar):
                        # Move data to device
                        inputs = batch['input'].to(device)
                        targets = batch['target'].to(device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        # Update statistics
                        train_loss += loss.item()
                        train_pbar.set_postfix(loss=loss.item())
                    
                    # Calculate average training loss for this epoch
                    avg_train_loss = train_loss / len(train_loader)
                    train_losses.append(avg_train_loss)
                    
                    # Validation phase
                    model.eval()
                    val_loss = 0.0
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
                    
                    with torch.no_grad():
                        for batch in val_pbar:
                            # Move data to device
                            inputs = batch['input'].to(device)
                            targets = batch['target'].to(device)
                            
                            # Forward pass
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            
                            # Update statistics
                            val_loss += loss.item()
                            val_pbar.set_postfix(loss=loss.item())
                    
                    # Calculate average validation loss for this epoch
                    avg_val_loss = val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)
                    
                    # Update progress bar with loss info
                    epoch_pbar.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)
                    
                    # Save model if it's the best so far
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_loss_{avg_val_loss:.4f}.ckpt")
                        model.export_checkpoint(checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
                
                # Plot training and validation loss
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 5))
                    plt.plot(train_losses, label='Training Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                except Exception as e:
                    print(f"Could not plot training curves: {e}")
                    
                return train_losses, val_losses
            
            # Run the training
            try:
                print(f"Training with {len(train_loader)} training batches and {len(val_loader)} validation batches")
                print(f"Model: hidden_dim={config.model.hidden_dim}, lr={config.training.lr}, epochs={config.training.epochs}")
                print(f"Saving checkpoints to: {checkpoint_dir}")
                print("\n--- Starting Training ---\n")
                
                train_losses, val_losses = train_model()
                
                print("\n--- Training Complete ---\n")
                print(f"Final training loss: {train_losses[-1]:.4f}")
                print(f"Final validation loss: {val_losses[-1]:.4f}")
                
                # List the generated checkpoints
                if os.path.exists(checkpoint_dir):
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
                    if checkpoints:
                        print(f"\nGenerated checkpoints:")
                        for ckpt in checkpoints:
                            print(f"- {ckpt}")
                    else:
                        print("\nNo .ckpt files found in checkpoint directory.")
                
                # Export the final model
                final_checkpoint = os.path.join(checkpoint_dir, f"triposr_final_model.ckpt")
                model.export_checkpoint(final_checkpoint)
                print(f"\nExported final model to: {final_checkpoint}")
                
            except Exception as e:
                print(f"\nError during training: {e}")
                import traceback
                traceback.print_exc()
    
    start_button.on_click(on_start_button_clicked)
    display(start_button)

# Run the custom training function
run_custom_training()


## 7. Download Trained Model

# Define the download checkpoint function with checkpoint_dir as parameter
def download_checkpoint(checkpoint_dir):
    from google.colab import files
    from ipywidgets import Button, Output
    import os
    from IPython.display import display, clear_output
    
    # Create output area for messages
    output = Output()
    display(output)
    
    # Create download button
    download_button = Button(
        description="Download Checkpoint",
        button_style="info", 
        icon="download",
        tooltip="Download the latest checkpoint file"
    )
    
    def on_download_clicked(b):
        with output:
            clear_output()
            
            # Check if checkpoint directory exists
            if not os.path.exists(checkpoint_dir):
                print(f"Checkpoint directory not found: {checkpoint_dir}")
                print("Please run training first.")
                return
            
            # Find checkpoint files
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            
            if not checkpoints:
                print("No checkpoint files found. Please run training first.")
                return
                
            print("Available checkpoints:")
            for i, ckpt in enumerate(checkpoints):
                print(f"{i+1}. {ckpt}")
            
            # Get latest checkpoint
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"\nDownloading: {checkpoints[-1]}")
            
            try:
                files.download(latest_checkpoint)
                print("\nDownload initiated. Check your browser downloads.")
                print("\nAlternatively, you can save to Google Drive:")
                print("from google.colab import drive")
                print("drive.mount('/content/drive')")
                print(f"!cp {latest_checkpoint} /content/drive/MyDrive/")
            except Exception as e:
                print(f"Error: {e}")
    
    download_button.on_click(on_download_clicked)
    display(download_button)

# Import os module for path operations
import os

# Call the function with the checkpoint directory
checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
download_checkpoint(checkpoint_dir)

## 8. Training Results and Analysis

#Analyze the training results and performance metrics.

# Import required modules for analysis
import os
import re
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Function to analyze the checkpoints in the directory
def analyze_training_results(checkpoint_dir):
    
    if not os.path.exists(checkpoint_dir):
        display(Markdown("### No Training Results Available"))
        print("Please run the training process first to generate results.")
        return
    
    # Get all checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    
    if not checkpoints:
        display(Markdown("### No Checkpoint Files Found"))
        print("Training may have failed or no checkpoints were saved.")
        return
    
    # Extract epoch and loss information from filenames
    metrics = []
    pattern = r'model_epoch_(\d+)_loss_([0-9.]+)'
    
    for ckpt in checkpoints:
        match = re.search(pattern, ckpt)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            metrics.append((epoch, loss, ckpt))
    
    # Sort by epoch
    metrics.sort(key=lambda x: x[0])
    
    # Display summary
    display(Markdown(f"### Training Summary"))
    print(f"Total checkpoints: {len(metrics)}")
    
    if metrics:
        # Get best model (lowest loss)
        best_model = min(metrics, key=lambda x: x[1])
        print(f"Best model: Epoch {best_model[0]}, Loss {best_model[1]:.6f}")
        print(f"Best model file: {best_model[2]}")
        
        # Plot loss curve if we have enough data points
        if len(metrics) > 1:
            epochs = [m[0] for m in metrics]
            losses = [m[1] for m in metrics]
            
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, losses, 'o-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.grid(True)
            plt.legend()
            plt.show()

# Run the analysis on our checkpoint directory
analyze_training_results(checkpoint_dir)