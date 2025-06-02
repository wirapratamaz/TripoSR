import os
import json
import yaml
import ipywidgets as widgets
from IPython.display import display
from omegaconf import OmegaConf
import torch


class ConfigManager:
    """
    Interactive configuration manager for TripoSR training.
    Provides a UI for configuring training parameters and saving/loading configurations.
    """
    
    def __init__(self, default_config_path=None, config_dict=None):
        """
        Initialize the configuration manager.
        
        Args:
            default_config_path: Path to default YAML config file
            config_dict: Dictionary with configuration values (alternative to file)
        """
        self.config = None
        self.config_widgets = {}
        self.sections = {}
        self.ui = None
        
        # Load configuration
        if default_config_path and os.path.exists(default_config_path):
            self.config = OmegaConf.load(default_config_path)
            print(f"Loaded configuration from {default_config_path}")
        elif config_dict:
            self.config = OmegaConf.create(config_dict)
            print("Loaded configuration from dictionary")
        else:
            # Create a default configuration
            self.config = self._create_default_config()
            print("Created default configuration")
    
    def _create_default_config(self):
        """Create a default configuration structure"""
        default_config = {
            'data': {
                'dataset_path': './dataset',
                'image_size': 224,
                'batch_size': 32,
                'num_workers': 4,
                'val_split': True
            },
            'model': {
                'name': 'triposr',
                'pretrained': True,
                'freeze_backbone': False
            },
            'train': {
                'epochs': 10,
                'batch_size': 32,
                'use_amp': True,
                'seed': 42
            },
            'optimizer': {
                'name': 'adam',
                'lr': 1e-4,
                'weight_decay': 1e-5
            },
            'scheduler': {
                'name': 'cosine',
                'warmup_epochs': 1,
                'min_lr': 1e-6
            },
            'val': {
                'batch_size': 32,
                'eval_global_steps': 100
            },
            'saver': {
                'checkpoint_dir': './checkpoints',
                'checkpoint_global_steps': 500
            },
            'logger': {
                'trackers': ['tensorboard'],
                'log_interval': 10
            }
        }
        
        return OmegaConf.create(default_config)
    
    def build_ui(self):
        """Build the configuration UI"""
        # Create tab structure for different config sections
        tabs = []
        tab_titles = []
        
        # Dataset configuration
        data_tab = self._create_data_section()
        tabs.append(data_tab)
        tab_titles.append('Data')
        
        # Model configuration
        model_tab = self._create_model_section()
        tabs.append(model_tab)
        tab_titles.append('Model')
        
        # Training configuration
        train_tab = self._create_training_section()
        tabs.append(train_tab)
        tab_titles.append('Training')
        
        # Optimizer configuration
        optim_tab = self._create_optimizer_section()
        tabs.append(optim_tab)
        tab_titles.append('Optimizer')
        
        # Validation configuration
        val_tab = self._create_validation_section()
        tabs.append(val_tab)
        tab_titles.append('Validation')
        
        # Saver configuration
        saver_tab = self._create_saver_section()
        tabs.append(saver_tab)
        tab_titles.append('Saver')
        
        # Logger configuration
        logger_tab = self._create_logger_section()
        tabs.append(logger_tab)
        tab_titles.append('Logger')
        
        # Create tab widget
        tab_widget = widgets.Tab()
        tab_widget.children = tabs
        
        # Set tab titles
        for i, title in enumerate(tab_titles):
            tab_widget.set_title(i, title)
        
        # Create buttons for save/load/apply
        save_button = widgets.Button(
            description='Save Config',
            button_style='primary',
            icon='save'
        )
        
        load_button = widgets.Button(
            description='Load Config',
            button_style='info',
            icon='upload'
        )
        
        apply_button = widgets.Button(
            description='Apply Changes',
            button_style='success',
            icon='check'
        )
        
        reset_button = widgets.Button(
            description='Reset to Default',
            button_style='warning',
            icon='refresh'
        )
        
        # Connect button events
        save_button.on_click(self._on_save_click)
        load_button.on_click(self._on_load_click)
        apply_button.on_click(self._on_apply_click)
        reset_button.on_click(self._on_reset_click)
        
        # Text field for config path
        self.config_path_input = widgets.Text(
            value='./openlrm_integration/configs/custom_config.yaml',
            description='Config Path:',
            style={'description_width': 'initial'}
        )
        
        # Output widget for messages
        self.output = widgets.Output()
        
        # Assemble final UI
        self.ui = widgets.VBox([
            widgets.HTML("<h2>TripoSR Training Configuration</h2>"),
            tab_widget,
            widgets.HBox([self.config_path_input]),
            widgets.HBox([save_button, load_button, apply_button, reset_button]),
            self.output
        ])
        
        # Update widgets with current config values
        self._update_widgets_from_config()
        
        return self.ui
    
    def _create_data_section(self):
        """Create widgets for data configuration"""
        # Dataset path
        dataset_path = widgets.Text(
            description='Dataset Path:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['data.dataset_path'] = dataset_path
        
        # Image size
        image_size = widgets.IntSlider(
            min=64,
            max=512,
            step=32,
            description='Image Size:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['data.image_size'] = image_size
        
        # Batch size
        batch_size = widgets.IntSlider(
            min=1,
            max=128,
            step=1,
            description='Batch Size:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['data.batch_size'] = batch_size
        
        # Number of workers
        num_workers = widgets.IntSlider(
            min=0,
            max=16,
            step=1,
            description='Num Workers:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['data.num_workers'] = num_workers
        
        # Validation split
        val_split = widgets.Checkbox(
            description='Use Validation Split',
            style={'description_width': 'initial'}
        )
        self.config_widgets['data.val_split'] = val_split
        
        # Assemble section
        section = widgets.VBox([
            widgets.HTML("<h3>Dataset Configuration</h3>"),
            dataset_path,
            image_size,
            batch_size,
            num_workers,
            val_split
        ])
        
        return section
    
    def _create_model_section(self):
        """Create widgets for model configuration"""
        # Model name
        model_name = widgets.Dropdown(
            options=['triposr', 'triposr-base', 'triposr-large'],
            description='Model:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['model.name'] = model_name
        
        # Pretrained
        pretrained = widgets.Checkbox(
            description='Use Pretrained Model',
            style={'description_width': 'initial'}
        )
        self.config_widgets['model.pretrained'] = pretrained
        
        # Freeze backbone
        freeze_backbone = widgets.Checkbox(
            description='Freeze Backbone',
            style={'description_width': 'initial'}
        )
        self.config_widgets['model.freeze_backbone'] = freeze_backbone
        
        # Assemble section
        section = widgets.VBox([
            widgets.HTML("<h3>Model Configuration</h3>"),
            model_name,
            pretrained,
            freeze_backbone
        ])
        
        return section
    
    def _create_training_section(self):
        """Create widgets for training configuration"""
        # Number of epochs
        epochs = widgets.IntSlider(
            min=1,
            max=100,
            step=1,
            description='Epochs:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['train.epochs'] = epochs
        
        # Batch size
        batch_size = widgets.IntSlider(
            min=1,
            max=128,
            step=1,
            description='Batch Size:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['train.batch_size'] = batch_size
        
        # Use AMP (Automatic Mixed Precision)
        use_amp = widgets.Checkbox(
            description='Use Mixed Precision (AMP)',
            style={'description_width': 'initial'}
        )
        self.config_widgets['train.use_amp'] = use_amp
        
        # Random seed
        seed = widgets.IntText(
            description='Random Seed:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['train.seed'] = seed
        
        # Assemble section
        section = widgets.VBox([
            widgets.HTML("<h3>Training Configuration</h3>"),
            epochs,
            batch_size,
            use_amp,
            seed
        ])
        
        return section
    
    def _create_optimizer_section(self):
        """Create widgets for optimizer configuration"""
        # Optimizer name
        optimizer_name = widgets.Dropdown(
            options=['adam', 'adamw', 'sgd'],
            description='Optimizer:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['optimizer.name'] = optimizer_name
        
        # Learning rate
        lr = widgets.FloatLogSlider(
            base=10,
            min=-6,  # 10^-6
            max=-2,  # 10^-2
            step=0.1,
            description='Learning Rate:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['optimizer.lr'] = lr
        
        # Weight decay
        weight_decay = widgets.FloatLogSlider(
            base=10,
            min=-8,  # 10^-8
            max=-4,  # 10^-4
            step=0.1,
            description='Weight Decay:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['optimizer.weight_decay'] = weight_decay
        
        # Scheduler
        scheduler_name = widgets.Dropdown(
            options=['cosine', 'step', 'none'],
            description='Scheduler:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['scheduler.name'] = scheduler_name
        
        # Warmup epochs
        warmup_epochs = widgets.IntSlider(
            min=0,
            max=10,
            step=1,
            description='Warmup Epochs:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['scheduler.warmup_epochs'] = warmup_epochs
        
        # Min learning rate
        min_lr = widgets.FloatLogSlider(
            base=10,
            min=-8,  # 10^-8
            max=-4,  # 10^-4
            step=0.1,
            description='Min LR:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['scheduler.min_lr'] = min_lr
        
        # Assemble section
        section = widgets.VBox([
            widgets.HTML("<h3>Optimizer Configuration</h3>"),
            optimizer_name,
            lr,
            weight_decay,
            widgets.HTML("<h4>Learning Rate Scheduler</h4>"),
            scheduler_name,
            warmup_epochs,
            min_lr
        ])
        
        return section
    
    def _create_validation_section(self):
        """Create widgets for validation configuration"""
        # Batch size
        batch_size = widgets.IntSlider(
            min=1,
            max=128,
            step=1,
            description='Batch Size:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['val.batch_size'] = batch_size
        
        # Evaluation frequency
        eval_steps = widgets.IntSlider(
            min=10,
            max=1000,
            step=10,
            description='Eval Frequency (steps):',
            style={'description_width': 'initial'}
        )
        self.config_widgets['val.eval_global_steps'] = eval_steps
        
        # Assemble section
        section = widgets.VBox([
            widgets.HTML("<h3>Validation Configuration</h3>"),
            batch_size,
            eval_steps
        ])
        
        return section
    
    def _create_saver_section(self):
        """Create widgets for saver configuration"""
        # Checkpoint directory
        checkpoint_dir = widgets.Text(
            description='Checkpoint Dir:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['saver.checkpoint_dir'] = checkpoint_dir
        
        # Checkpoint frequency
        checkpoint_steps = widgets.IntSlider(
            min=100,
            max=5000,
            step=100,
            description='Checkpoint Frequency (steps):',
            style={'description_width': 'initial'}
        )
        self.config_widgets['saver.checkpoint_global_steps'] = checkpoint_steps
        
        # Assemble section
        section = widgets.VBox([
            widgets.HTML("<h3>Checkpoint Configuration</h3>"),
            checkpoint_dir,
            checkpoint_steps
        ])
        
        return section
    
    def _create_logger_section(self):
        """Create widgets for logger configuration"""
        # Logger selection
        trackers = widgets.SelectMultiple(
            options=['tensorboard', 'wandb', 'csv'],
            value=['tensorboard'],
            description='Loggers:',
            style={'description_width': 'initial'}
        )
        self.config_widgets['logger.trackers'] = trackers
        
        # Log interval
        log_interval = widgets.IntSlider(
            min=1,
            max=100,
            step=1,
            description='Log Interval (steps):',
            style={'description_width': 'initial'}
        )
        self.config_widgets['logger.log_interval'] = log_interval
        
        # Assemble section
        section = widgets.VBox([
            widgets.HTML("<h3>Logging Configuration</h3>"),
            trackers,
            log_interval
        ])
        
        return section
    
    def _update_widgets_from_config(self):
        """Update widget values from the current configuration"""
        for path, widget in self.config_widgets.items():
            # Get the value from config
            keys = path.split('.')
            value = self.config
            for key in keys:
                if key in value:
                    value = value[key]
                else:
                    # Key doesn't exist in config
                    value = None
                    break
            
            # Skip if value is None
            if value is None:
                continue
            
            # Set widget value based on widget type
            if isinstance(widget, widgets.Text) or isinstance(widget, widgets.IntText):
                widget.value = value
            elif isinstance(widget, widgets.IntSlider) or isinstance(widget, widgets.FloatSlider):
                widget.value = value
            elif isinstance(widget, widgets.FloatLogSlider):
                widget.value = value
            elif isinstance(widget, widgets.Checkbox):
                widget.value = value
            elif isinstance(widget, widgets.Dropdown):
                if value in widget.options:
                    widget.value = value
            elif isinstance(widget, widgets.SelectMultiple):
                # Convert to list if it's not already
                if not isinstance(value, list):
                    value = [value]
                # Set only values that are in options
                widget.value = [v for v in value if v in widget.options]
    
    def _update_config_from_widgets(self):
        """Update configuration from widget values"""
        for path, widget in self.config_widgets.items():
            # Get the keys
            keys = path.split('.')
            
            # Get the widget value
            if isinstance(widget, widgets.SelectMultiple):
                value = list(widget.value)
            else:
                value = widget.value
            
            # Update config
            config_dict = self.config
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    # Last key, set the value
                    config_dict[key] = value
                else:
                    # Create nested dict if needed
                    if key not in config_dict:
                        config_dict[key] = {}
                    config_dict = config_dict[key]
    
    def _on_save_click(self, b):
        """Handle save button click"""
        # Update config from widgets
        self._update_config_from_widgets()
        
        # Get config path
        config_path = self.config_path_input.value
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(self.config), f, default_flow_style=False)
        
        # Show message
        with self.output:
            self.output.clear_output()
            print(f"Configuration saved to {config_path}")
    
    def _on_load_click(self, b):
        """Handle load button click"""
        # Get config path
        config_path = self.config_path_input.value
        
        # Check if file exists
        if not os.path.exists(config_path):
            with self.output:
                self.output.clear_output()
                print(f"Configuration file not found: {config_path}")
            return
        
        # Load config
        self.config = OmegaConf.load(config_path)
        
        # Update widgets
        self._update_widgets_from_config()
        
        # Show message
        with self.output:
            self.output.clear_output()
            print(f"Configuration loaded from {config_path}")
    
    def _on_apply_click(self, b):
        """Handle apply button click"""
        # Update config from widgets
        self._update_config_from_widgets()
        
        # Show message
        with self.output:
            self.output.clear_output()
            print("Configuration updated")
            print("\nCurrent configuration:")
            print(OmegaConf.to_yaml(self.config))
    
    def _on_reset_click(self, b):
        """Handle reset button click"""
        # Reset to default config
        self.config = self._create_default_config()
        
        # Update widgets
        self._update_widgets_from_config()
        
        # Show message
        with self.output:
            self.output.clear_output()
            print("Configuration reset to default")
    
    def get_config(self):
        """Get the current configuration"""
        # Update config from widgets if UI is created
        if self.ui is not None:
            self._update_config_from_widgets()
        
        return self.config
    
    def display(self):
        """Display the configuration UI"""
        if self.ui is None:
            self.build_ui()
        
        display(self.ui)


# Example usage:
"""
# Create and display the configuration manager
config_manager = ConfigManager(default_config_path='path/to/default/config.yaml')
config_manager.display()

# Later, get the configuration
config = config_manager.get_config()
"""
