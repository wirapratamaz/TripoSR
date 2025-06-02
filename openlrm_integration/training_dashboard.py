import torch
import time
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display

from .visible_training_loop import VisibleTrainingLoop
from .metrics_visualization import MetricsVisualizer

class TrainingDashboard:
    """
    Integrated training dashboard for TripoSR that combines the visible training
    loop with enhanced metrics visualization.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader=None,
        device='cuda',
        epochs=10,
        save_dir='./checkpoints',
        export_ckpt=True,
        log_interval=10,
        use_amp=True,
        metric_names=None,
        dark_mode=False
    ):
        """
        Initialize the training dashboard.
        
        Args:
            model: The PyTorch model to train
            optimizer: The optimizer to use
            criterion: The loss function
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            device: Device to train on ('cuda' or 'cpu')
            epochs: Number of epochs to train for
            save_dir: Directory to save checkpoints
            export_ckpt: Whether to export .ckpt files (compatible with OpenLRM)
            log_interval: How often to log metrics (in batches)
            use_amp: Whether to use automatic mixed precision
            metric_names: List of metric names to track beyond loss
            dark_mode: Whether to use dark mode for plots
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.export_ckpt = export_ckpt
        self.log_interval = log_interval
        self.use_amp = use_amp
        
        # Initialize training loop and metrics visualizer
        self.training_loop = VisibleTrainingLoop(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            save_dir=save_dir,
            export_ckpt=export_ckpt,
            log_interval=log_interval,
            use_amp=use_amp
        )
        
        self.metrics_visualizer = MetricsVisualizer(
            metric_names=metric_names,
            dark_mode=dark_mode
        )
        
        # Initialize UI elements
        self.dashboard_ui = None
        self.step_counter = 0
        self.epoch_counter = 0
        
    def initialize_ui(self):
        """Initialize the dashboard UI"""
        # Initialize metrics visualizer dashboard
        self.metrics_visualizer.initialize_dashboard()
        
        # Create additional UI elements
        self.control_panel = self._create_control_panel()
        
        # Display the full dashboard
        self.dashboard_ui = widgets.VBox([
            widgets.HTML("<h1 style='text-align: center;'>TripoSR Training Dashboard</h1>"),
            self.control_panel,
            widgets.HTML("<hr style='margin: 20px 0;'>")
        ])
        
        display(self.dashboard_ui)
    
    def _create_control_panel(self):
        """Create control panel UI elements"""
        # Learning rate slider
        lr_slider = widgets.FloatLogSlider(
            value=self.optimizer.param_groups[0]['lr'],
            base=10,
            min=-5,  # 10^-5
            max=-1,  # 10^-1
            step=0.1,
            description='Learning Rate:',
            continuous_update=False
        )
        
        # Connect slider to optimizer
        def on_lr_change(change):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = change['new']
            print(f"Learning rate changed to {change['new']:.6f}")
        
        lr_slider.observe(on_lr_change, names='value')
        
        # Create buttons for common actions
        save_button = widgets.Button(
            description='Save Checkpoint',
            button_style='primary',
            icon='save'
        )
        
        def on_save_button_click(b):
            path = f"{self.save_dir}/manual_checkpoint.pt"
            self._save_checkpoint(path)
            print(f"Manual checkpoint saved to {path}")
        
        save_button.on_click(on_save_button_click)
        
        # Visualization options
        dark_mode_toggle = widgets.Checkbox(
            value=self.metrics_visualizer.dark_mode,
            description='Dark Mode',
            indent=False
        )
        
        def on_dark_mode_toggle(change):
            self.metrics_visualizer.dark_mode = change['new']
            self.metrics_visualizer.setup_style()
            self.metrics_visualizer.update_dashboard(force=True)
        
        dark_mode_toggle.observe(on_dark_mode_toggle, names='value')
        
        # GPU memory monitor
        gpu_monitor = widgets.HTML("GPU Memory: Initializing...")
        
        # Create a control panel with all elements
        control_panel = widgets.VBox([
            widgets.HTML("<h3>Training Controls</h3>"),
            widgets.HBox([lr_slider, save_button]),
            widgets.HBox([dark_mode_toggle, gpu_monitor])
        ])
        
        # Update GPU memory periodically
        def update_gpu_memory():
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                gpu_monitor.value = (
                    f"GPU Memory: {memory_allocated:.1f}MB allocated, "
                    f"{memory_reserved:.1f}MB reserved, "
                    f"{memory_allocated/total_memory*100:.1f}% of {total_memory:.1f}MB total"
                )
            else:
                gpu_monitor.value = "GPU Memory: No GPU available"
        
        update_gpu_memory()  # Initial update
        
        # Return the control panel
        return control_panel
    
    def _save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch_counter,
            'step': self.step_counter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, path)
        
        # Save metrics history
        metrics_path = path.replace('.pt', '_metrics.json')
        self.metrics_visualizer.save_metric_history(metrics_path)
    
    def train(self):
        """
        Run the full training process with the enhanced dashboard.
        This method combines the visible training loop with the enhanced
        metrics visualization.
        """
        print(f"Starting training for {self.epochs} epochs")
        print(f"Using device: {self.device}")
        print(f"Mixed precision: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Initialize the UI
        self.initialize_ui()
        
        # Setup scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        # Main training loop with progress bar for epochs
        for epoch in tqdm(range(self.epochs), desc="Training Progress", unit="epoch"):
            self.epoch_counter = epoch
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            samples_seen = 0
            
            # Progress bar for batches
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False,
                unit="batch"
            )
            
            for i, batch in enumerate(pbar):
                self.step_counter += 1
                batch_start_time = time.time()
                
                # Move data to device
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    # Handle case where batch is a dictionary
                    inputs = batch
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            inputs[k] = inputs[k].to(self.device)
                    targets = None
                
                # Forward pass with optional mixed precision
                if self.use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        if targets is not None:
                            loss = self.criterion(outputs, targets)
                        else:
                            # Handle case where model returns loss directly
                            loss = outputs if isinstance(outputs, torch.Tensor) else outputs['loss']
                    
                    # Backward and optimize with gradient scaling
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Standard training without mixed precision
                    outputs = self.model(inputs)
                    if targets is not None:
                        loss = self.criterion(outputs, targets)
                    else:
                        # Handle case where model returns loss directly
                        loss = outputs if isinstance(outputs, torch.Tensor) else outputs['loss']
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                train_loss += batch_loss
                
                # Batch size might vary in the last batch
                batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else inputs[list(inputs.keys())[0]].size(0)
                samples_seen += batch_size
                
                # Calculate batch processing time
                batch_time = time.time() - batch_start_time
                throughput = batch_size / batch_time
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'avg_loss': f"{train_loss / (i + 1):.4f}",
                    'samples/sec': f"{throughput:.1f}"
                })
                
                # Update metrics visualizer
                metrics = {
                    'loss': batch_loss,
                    'batch_time': batch_time,
                    'throughput': throughput,
                    'gpu_memory': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                }
                
                self.metrics_visualizer.add_train_metrics(metrics, step=self.step_counter, epoch=epoch)
                
                # Update dashboard periodically
                if i % self.log_interval == 0 or i == len(self.train_loader) - 1:
                    self.metrics_visualizer.update_dashboard()
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_loss = 0.0
            if self.val_loader:
                self.model.eval()
                val_start_time = time.time()
                
                with torch.no_grad():
                    for val_batch in tqdm(self.val_loader, desc=f"Validation {epoch+1}", leave=False):
                        # Move data to device
                        if isinstance(val_batch, (list, tuple)) and len(val_batch) == 2:
                            val_inputs, val_targets = val_batch
                            val_inputs = val_inputs.to(self.device)
                            val_targets = val_targets.to(self.device)
                        else:
                            # Handle case where batch is a dictionary
                            val_inputs = val_batch
                            for k in val_inputs:
                                if isinstance(val_inputs[k], torch.Tensor):
                                    val_inputs[k] = val_inputs[k].to(self.device)
                            val_targets = None
                        
                        # Forward pass
                        val_outputs = self.model(val_inputs)
                        if val_targets is not None:
                            val_batch_loss = self.criterion(val_outputs, val_targets)
                        else:
                            # Handle case where model returns loss directly
                            val_batch_loss = val_outputs if isinstance(val_outputs, torch.Tensor) else val_outputs['loss']
                        
                        val_loss += val_batch_loss.item()
                
                avg_val_loss = val_loss / len(self.val_loader)
                val_time = time.time() - val_start_time
            else:
                avg_val_loss = None
            
            # Add epoch metrics
            epoch_train_metrics = {'loss': avg_train_loss}
            epoch_val_metrics = {'loss': avg_val_loss} if avg_val_loss is not None else None
            
            self.metrics_visualizer.add_epoch_metrics(epoch_train_metrics, epoch_val_metrics, epoch)
            
            # Update dashboard with epoch results
            self.metrics_visualizer.update_dashboard(force=True)
            
            # Save checkpoint
            checkpoint_path = f"{self.save_dir}/checkpoint_epoch_{epoch+1}.pt"
            self._save_checkpoint(checkpoint_path)
            
            # Export in OpenLRM .ckpt format if requested
            if self.export_ckpt:
                ckpt_path = f"{self.save_dir}/model_epoch_{epoch+1}.ckpt"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Exported model to {ckpt_path}")
            
            # Log epoch results
            print(f"\nEpoch {epoch+1}/{self.epochs} completed:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            if avg_val_loss is not None:
                print(f"  Validation Loss: {avg_val_loss:.6f}")
            print(f"  Training Speed: {samples_seen / epoch_time:.2f} samples/sec")
            print(f"  Epoch Time: {epoch_time:.2f} sec")
        
        print("\nTraining completed!")
        
        # Generate final metrics report
        self.metrics_visualizer.create_metrics_report(f"{self.save_dir}/training_report.png")
        
        return self.model


# Example usage in notebook:
"""
# Import the necessary classes
from openlrm_integration.training_dashboard import TrainingDashboard

# Create model, optimizer, criterion, dataloaders as usual
model = YourTripoSRModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = YourLossFunction()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create the training dashboard
dashboard = TrainingDashboard(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    save_dir='./checkpoints',
    export_ckpt=True,
    metric_names=['loss', 'psnr', 'ssim'],  # Add your specific metrics
    dark_mode=False  # Set to True for dark mode
)

# Start training with the dashboard
trained_model = dashboard.train()
"""
