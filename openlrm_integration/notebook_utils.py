import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_file_upload(dataset_path):
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
    
    if not in_colab:
        return
    
    # Create upload button
    upload_button = widgets.Button(
        description='Upload .glb Files',
        button_style='primary',
        tooltip='Click to upload .glb files for training',
        icon='upload'
    )
    
    output = widgets.Output()
    
    def on_upload_button_clicked(b):
        with output:
            output.clear_output()
            print("Please select .glb files to upload...")
            uploaded = files.upload()
            
            # Process uploaded files
            for filename, content in uploaded.items():
                if not filename.endswith('.glb'):
                    print(f"Skipping {filename} - not a .glb file")
                    continue
                
                # Determine split (80% train, 20% val)
                import random
                split = "train" if random.random() < 0.8 else "val"
                
                # Save the file
                target_path = os.path.join(dataset_path, split, filename)
                with open(target_path, 'wb') as f:
                    f.write(content)
                
                print(f"Saved {filename} to {target_path}")
            
            print(f"\nUploaded {len(uploaded)} files")
            print(f"Training files: {len(os.listdir(os.path.join(dataset_path, 'train')))}")
            print(f"Validation files: {len(os.listdir(os.path.join(dataset_path, 'val')))}")
    
    upload_button.on_click(on_upload_button_clicked)
    
    # Display upload button
    display(widgets.VBox([
        widgets.HTML("<h3>Upload .glb Files for Training</h3>"),
        upload_button,
        output
    ]))
    
    # Add option to use sample data for testing
    use_sample_data = widgets.Checkbox(
        value=False,
        description='Use sample data for testing',
        disabled=False
    )
    
    sample_data_output = widgets.Output()
    
    def on_sample_data_change(change):
        if change['new'] == True:
            with sample_data_output:
                sample_data_output.clear_output()
                
                # Download sample .glb files
                import urllib.request
                import shutil
                
                sample_urls = [
                    "https://market-assets.fra1.cdn.digitaloceanspaces.com/market-assets/samples/Astronaut.glb",
                    "https://market-assets.fra1.cdn.digitaloceanspaces.com/market-assets/samples/DamagedHelmet.glb"
                ]
                
                for i, url in enumerate(sample_urls):
                    filename = url.split('/')[-1]
                    print(f"Downloading {filename}...")
                    
                    # Determine split (first to train, second to val for testing both)
                    split = "train" if i % 2 == 0 else "val"
                    target_path = os.path.join(dataset_path, split, filename)
                    
                    try:
                        with urllib.request.urlopen(url) as response, open(target_path, 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)
                        print(f"Saved to {target_path}")
                    except Exception as e:
                        print(f"Error downloading {filename}: {e}")
                
                print(f"\nDownloaded sample files")
                print(f"Training files: {len(os.listdir(os.path.join(dataset_path, 'train')))}")
                print(f"Validation files: {len(os.listdir(os.path.join(dataset_path, 'val')))}")
    
    use_sample_data.observe(on_sample_data_change, names='value')
    
    display(widgets.VBox([
        widgets.HTML("<h3>Or Use Sample Data</h3>"),
        use_sample_data,
        sample_data_output
    ]))

def export_to_ckpt(model, save_path, config=None):
    """
    Export model to OpenLRM .ckpt format and provide download in Colab.
    
    Args:
        model: The model to export
        save_path: Path where to save the .ckpt file
        config: Optional config to include in the checkpoint
    
    Returns:
        Path to the saved checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Check if we're running with OpenLRM
        if hasattr(model, 'export_checkpoint'):
            # Use the native OpenLRM export functionality
            model.export_checkpoint(save_path)
            print(f"Model exported to OpenLRM format at {save_path}")
        else:
            # Manual export to similar format
            import datetime
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': config,
                'metadata': {
                    'export_time': datetime.datetime.now().isoformat(),
                    'format_version': '1.0'
                }
            }
            
            torch.save(checkpoint, save_path)
            print(f"Model exported to checkpoint format at {save_path}")
        
        # Check if running in Colab to provide download link
        try:
            import google.colab
            from google.colab import files
            
            print(f"Downloading {save_path}...")
            files.download(save_path)
        except ImportError:
            print(f"File saved at: {save_path}")
            print("To download, use your file browser to access this location.")
        
        return save_path
            
    except Exception as e:
        print(f"Error exporting model: {e}")
        return None

def create_export_button(model, checkpoint_dir, has_openlrm=False):
    """
    Create a button to export the model to .ckpt format.
    
    Args:
        model: The model to export
        checkpoint_dir: Directory to save the checkpoint
        has_openlrm: Whether OpenLRM is available
    
    Returns:
        None
    """
    import ipywidgets as widgets
    from IPython.display import display
    
    export_button = widgets.Button(
        description='Export to .ckpt',
        button_style='success',
        tooltip='Export model to OpenLRM .ckpt format',
        icon='download'
    )
    
    export_output = widgets.Output()
    
    def on_export_button_clicked(b):
        with export_output:
            export_output.clear_output()
            
            import datetime
            export_path = os.path.join(
                checkpoint_dir, 
                f"triposr_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt"
            )
            
            export_to_ckpt(model, export_path)
    
    export_button.on_click(on_export_button_clicked)
    
    # Display export button
    display(widgets.VBox([
        widgets.HTML("<h3>Export and Download OpenLRM Format Checkpoint</h3>"),
        export_button,
        export_output
    ]))

def show_training_progress(epoch, total_epochs, batch, total_batches, loss, metrics=None):
    """
    Display training progress with tqdm progress bars.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        batch: Current batch
        total_batches: Total number of batches
        loss: Current loss value
        metrics: Optional dict of additional metrics to display
    
    Returns:
        None
    """
    # Update progress bars and display metrics
    epoch_bar = tqdm(total=total_epochs, desc="Training Progress", position=0, leave=True)
    epoch_bar.update(epoch)
    
    batch_bar = tqdm(total=total_batches, desc=f"Epoch {epoch}/{total_epochs}", position=1, leave=False)
    batch_bar.update(batch)
    
    # Display loss and metrics
    metrics_str = f"Loss: {loss:.6f}"
    if metrics:
        for name, value in metrics.items():
            metrics_str += f", {name}: {value:.6f}"
    
    print(metrics_str)
    
    return epoch_bar, batch_bar

# Example usage in notebook:
"""
# Import the utilities
from openlrm_integration.notebook_utils import setup_file_upload, export_to_ckpt, create_export_button, show_training_progress

# Set up file upload
setup_file_upload('./dataset')

# After training is complete
create_export_button(model, './checkpoints')
"""
