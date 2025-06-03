import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import display, clear_output
import ipywidgets as widgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from PIL import Image


class MetricsVisualizer:
    """
    Enhanced metrics visualization for TripoSR training.
    Provides real-time, interactive visualizations of training metrics.
    """
    
    def __init__(self, metric_names=None, max_history=1000, dark_mode=False):
        """
        Initialize the metrics visualizer.
        
        Args:
            metric_names: List of metric names to track beyond loss (e.g., ['psnr', 'ssim'])
            max_history: Maximum number of data points to keep in history
            dark_mode: Whether to use dark mode for plots
        """
        self.metric_names = metric_names or ['loss']
        if 'loss' not in self.metric_names:
            self.metric_names = ['loss'] + self.metric_names
            
        self.max_history = max_history
        self.dark_mode = dark_mode
        
        # Initialize metric tracking
        self.train_metrics = {name: [] for name in self.metric_names}
        self.val_metrics = {name: [] for name in self.metric_names}
        self.epoch_train_metrics = {name: [] for name in self.metric_names}
        self.epoch_val_metrics = {name: [] for name in self.metric_names}
        
        # Performance metrics
        self.batch_times = []
        self.throughputs = []
        self.gpu_memory_usage = []
        self.epochs_completed = []
        
        # Initialize figures
        self.figures = {}
        self.setup_style()
        
        # Interactive widgets
        self.output_widget = widgets.Output()
        self.dashboard = None
        
        # Track current step and epoch
        self.current_step = 0
        self.current_epoch = 0
        
    def setup_style(self):
        """Configure plot style based on mode"""
        if self.dark_mode:
            plt.style.use('dark_background')
            self.bg_color = '#121212'
            self.text_color = 'white'
            self.grid_color = '#333333'
            self.line_colors = ['#00bfff', '#ff4500', '#00ff7f', '#ffd700', '#ff69b4']
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = 'white'
            self.text_color = 'black'
            self.grid_color = '#cccccc'
            self.line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def initialize_dashboard(self):
        """Initialize the interactive dashboard"""
        self.dashboard = widgets.VBox([
            widgets.HTML("<h2 style='text-align: center;'>TripoSR Training Metrics Dashboard</h2>"),
            self.output_widget
        ])
        display(self.dashboard)
        
    def add_train_metrics(self, metrics_dict, step, epoch):
        """
        Add training metrics for the current step.
        
        Args:
            metrics_dict: Dictionary of metric values
            step: Current step
            epoch: Current epoch
        """
        self.current_step = step
        self.current_epoch = epoch
        
        # Add metrics to history
        for name in self.metric_names:
            if name in metrics_dict:
                self.train_metrics[name].append(metrics_dict[name])
                # Trim history if needed
                if len(self.train_metrics[name]) > self.max_history:
                    self.train_metrics[name] = self.train_metrics[name][-self.max_history:]
        
        # Add performance metrics
        if 'batch_time' in metrics_dict:
            self.batch_times.append(metrics_dict['batch_time'])
            if len(self.batch_times) > self.max_history:
                self.batch_times = self.batch_times[-self.max_history:]
                
        if 'throughput' in metrics_dict:
            self.throughputs.append(metrics_dict['throughput'])
            if len(self.throughputs) > self.max_history:
                self.throughputs = self.throughputs[-self.max_history:]
                
        if 'gpu_memory' in metrics_dict:
            self.gpu_memory_usage.append(metrics_dict['gpu_memory'])
            if len(self.gpu_memory_usage) > self.max_history:
                self.gpu_memory_usage = self.gpu_memory_usage[-self.max_history:]
    
    def add_epoch_metrics(self, train_metrics, val_metrics=None, epoch=None):
        """
        Add metrics for a completed epoch.
        
        Args:
            train_metrics: Dictionary of training metrics for the epoch
            val_metrics: Dictionary of validation metrics for the epoch
            epoch: Epoch number
        """
        if epoch is not None:
            self.current_epoch = epoch
            self.epochs_completed.append(epoch)
        
        # Add training metrics
        for name in self.metric_names:
            if name in train_metrics:
                self.epoch_train_metrics[name].append(train_metrics[name])
        
        # Add validation metrics
        if val_metrics:
            for name in self.metric_names:
                if name in val_metrics:
                    self.epoch_val_metrics[name].append(val_metrics[name])
    
    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0
        
        # Get current GPU memory usage
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)
    
    def update_dashboard(self, force=False):
        """Update the dashboard with the latest metrics"""
        # Only update periodically to avoid excessive updates
        if not force and self.current_step % 10 != 0:
            return
            
        with self.output_widget:
            clear_output(wait=True)
            
            # Create a 2x3 grid of plots
            fig = plt.figure(figsize=(18, 12))
            
            # 1. Training Loss
            ax1 = plt.subplot(2, 3, 1)
            self._plot_training_metrics(ax1)
            
            # 2. Epoch Metrics
            ax2 = plt.subplot(2, 3, 2)
            self._plot_epoch_metrics(ax2)
            
            # 3. GPU Memory Usage
            ax3 = plt.subplot(2, 3, 3)
            self._plot_gpu_memory(ax3)
            
            # 4. Training Throughput
            ax4 = plt.subplot(2, 3, 4)
            self._plot_throughput(ax4)
            
            # 5. Learning Rate
            ax5 = plt.subplot(2, 3, 5)
            self._plot_learning_rate(ax5)
            
            # 6. Additional Custom Metrics or 3D Preview
            ax6 = plt.subplot(2, 3, 6)
            self._plot_custom_metrics(ax6)
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            
            # Add summary statistics below the plots
            self._display_summary_stats()
    
    def _plot_training_metrics(self, ax):
        """Plot training metrics over steps"""
        ax.set_title('Training Metrics (Live)', fontsize=12)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        
        for i, name in enumerate(self.metric_names):
            if self.train_metrics[name]:
                ax.plot(
                    self.train_metrics[name], 
                    label=name, 
                    color=self.line_colors[i % len(self.line_colors)]
                )
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_epoch_metrics(self, ax):
        """Plot metrics over epochs"""
        ax.set_title('Metrics by Epoch', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        
        x = list(range(1, len(self.epoch_train_metrics['loss']) + 1))
        
        for i, name in enumerate(self.metric_names):
            if self.epoch_train_metrics[name]:
                ax.plot(
                    x,
                    self.epoch_train_metrics[name],
                    label=f'Train {name}',
                    color=self.line_colors[i % len(self.line_colors)],
                    marker='o'
                )
            
            if self.epoch_val_metrics[name]:
                ax.plot(
                    x,
                    self.epoch_val_metrics[name],
                    label=f'Val {name}',
                    color=self.line_colors[i % len(self.line_colors)],
                    linestyle='--',
                    marker='x'
                )
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_gpu_memory(self, ax):
        """Plot GPU memory usage over time"""
        ax.set_title('GPU Memory Usage', fontsize=12)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Memory (MB)')
        
        # Add current memory usage
        current_memory = self._get_gpu_memory_usage()
        self.gpu_memory_usage.append(current_memory)
        
        if len(self.gpu_memory_usage) > self.max_history:
            self.gpu_memory_usage = self.gpu_memory_usage[-self.max_history:]
        
        if self.gpu_memory_usage:
            ax.plot(self.gpu_memory_usage, color=self.line_colors[0])
            
            # Add horizontal line for current memory
            ax.axhline(
                y=current_memory,
                color='r',
                linestyle='--',
                alpha=0.6,
                label=f'Current: {current_memory:.1f} MB'
            )
            
            # Show total GPU memory as a reference
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                ax.axhline(
                    y=total_memory,
                    color='gray',
                    linestyle=':',
                    alpha=0.5,
                    label=f'Total: {total_memory:.1f} MB'
                )
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_throughput(self, ax):
        """Plot training throughput over time"""
        ax.set_title('Training Throughput', fontsize=12)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Samples/sec')
        
        if self.throughputs:
            ax.plot(self.throughputs, color=self.line_colors[1])
            
            # Add moving average
            window_size = min(20, len(self.throughputs))
            if window_size > 0:
                moving_avg = np.convolve(
                    self.throughputs, 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                ax.plot(
                    range(window_size-1, len(self.throughputs)), 
                    moving_avg,
                    color=self.line_colors[2],
                    linestyle='--',
                    label='Moving Avg'
                )
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_learning_rate(self, ax):
        """Plot learning rate over epochs if available"""
        ax.set_title('Learning Rate', fontsize=12)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        
        # This is a placeholder - actual LR tracking would need to be implemented
        # in the training loop and passed to this method
        ax.text(
            0.5, 0.5, 
            'Learning Rate Tracking\n(Implement in Training Loop)', 
            ha='center', va='center',
            transform=ax.transAxes
        )
        
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_custom_metrics(self, ax):
        """Plot any additional custom metrics"""
        # This can be customized based on the specific metrics
        # that are important for TripoSR training
        ax.set_title('Custom Model Metrics', fontsize=12)
        
        # Placeholder for custom metrics
        ax.text(
            0.5, 0.5, 
            'Additional Model-Specific Metrics\n(Customize as needed)', 
            ha='center', va='center',
            transform=ax.transAxes
        )
        
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _display_summary_stats(self):
        """Display summary statistics below the plots"""
        # Create a nicely formatted HTML table for statistics
        html = """
        <style>
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-family: Arial, sans-serif;
        }
        .stats-table th, .stats-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .stats-table th {
            background-color: #f2f2f2;
            color: #333;
        }
        .stats-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .stats-header {
            font-size: 16px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        </style>
        
        <div class="stats-header">Training Summary Statistics</div>
        <table class="stats-table">
            <tr>
                <th>Metric</th>
                <th>Current Value</th>
                <th>Epoch Average</th>
                <th>Best Value</th>
            </tr>
        """
        
        # Add rows for each metric
        for name in self.metric_names:
            current = self.train_metrics[name][-1] if self.train_metrics[name] else "N/A"
            epoch_avg = np.mean(self.epoch_train_metrics[name]) if self.epoch_train_metrics[name] else "N/A"
            
            # For loss, best is minimum; for others like accuracy, best is maximum
            if name == 'loss':
                best = np.min(self.epoch_train_metrics[name]) if self.epoch_train_metrics[name] else "N/A"
                best_label = "Min"
            else:
                best = np.max(self.epoch_train_metrics[name]) if self.epoch_train_metrics[name] else "N/A"
                best_label = "Max"
            
            html += f"""
            <tr>
                <td>{name.capitalize()}</td>
                <td>{current if isinstance(current, str) else f"{current:.6f}"}</td>
                <td>{epoch_avg if isinstance(epoch_avg, str) else f"{epoch_avg:.6f}"}</td>
                <td>{best_label}: {best if isinstance(best, str) else f"{best:.6f}"}</td>
            </tr>
            """
        
        # Add performance metrics
        current_throughput = self.throughputs[-1] if self.throughputs else "N/A"
        avg_throughput = np.mean(self.throughputs) if self.throughputs else "N/A"
        
        html += f"""
        <tr>
            <td>Throughput (samples/sec)</td>
            <td>{current_throughput if isinstance(current_throughput, str) else f"{current_throughput:.2f}"}</td>
            <td>{avg_throughput if isinstance(avg_throughput, str) else f"{avg_throughput:.2f}"}</td>
            <td>Max: {np.max(self.throughputs) if self.throughputs else "N/A":.2f}</td>
        </tr>
        <tr>
            <td>GPU Memory (MB)</td>
            <td>{self._get_gpu_memory_usage():.2f}</td>
            <td>{np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else "N/A":.2f}</td>
            <td>Peak: {np.max(self.gpu_memory_usage) if self.gpu_memory_usage else "N/A":.2f}</td>
        </tr>
        <tr>
            <td>Training Progress</td>
            <td>Step {self.current_step}</td>
            <td>Epoch {self.current_epoch}</td>
            <td>Completed: {len(self.epochs_completed)} epochs</td>
        </tr>
        """
        
        html += """
        </table>
        """
        
        display(widgets.HTML(html))
        
    def create_metrics_report(self, save_path=None):
        """
        Create a comprehensive training metrics report.
        
        Args:
            save_path: Path to save the report (if None, just displays it)
        """
        report_fig = plt.figure(figsize=(20, 15))
        
        # More detailed plots for the report
        # 1. Training and Validation Loss
        ax1 = plt.subplot(3, 2, 1)
        self._plot_epoch_metrics(ax1)
        
        # 2. Additional Metrics (if available)
        ax2 = plt.subplot(3, 2, 2)
        self._plot_custom_metrics(ax2)
        
        # 3. Training Throughput
        ax3 = plt.subplot(3, 2, 3)
        self._plot_throughput(ax3)
        
        # 4. GPU Memory Usage
        ax4 = plt.subplot(3, 2, 4)
        self._plot_gpu_memory(ax4)
        
        # 5. Training vs Validation Comparison
        ax5 = plt.subplot(3, 2, 5)
        self._plot_train_val_comparison(ax5)
        
        # 6. Learning Curve Analysis
        ax6 = plt.subplot(3, 2, 6)
        self._plot_learning_curve_analysis(ax6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Metrics report saved to {save_path}")
        
        display(report_fig)
        plt.close(report_fig)
        
    def _plot_train_val_comparison(self, ax):
        """Plot comparison between training and validation metrics"""
        ax.set_title('Training vs Validation Comparison', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        
        x = list(range(1, len(self.epoch_train_metrics['loss']) + 1))
        
        if self.epoch_train_metrics['loss']:
            ax.plot(
                x, 
                self.epoch_train_metrics['loss'], 
                'b-', 
                label='Training Loss',
                marker='o'
            )
        
        if self.epoch_val_metrics['loss']:
            ax.plot(
                x, 
                self.epoch_val_metrics['loss'], 
                'r-', 
                label='Validation Loss',
                marker='x'
            )
            
            # Calculate and show gap between training and validation
            if self.epoch_train_metrics['loss'] and len(self.epoch_train_metrics['loss']) == len(self.epoch_val_metrics['loss']):
                gaps = np.array(self.epoch_val_metrics['loss']) - np.array(self.epoch_train_metrics['loss'])
                ax.fill_between(
                    x,
                    self.epoch_train_metrics['loss'],
                    self.epoch_val_metrics['loss'],
                    color='gray',
                    alpha=0.2,
                    label='Generalization Gap'
                )
                
                # Add text annotation for the current gap
                if gaps.size > 0:
                    current_gap = gaps[-1]
                    ax.annotate(
                        f'Current Gap: {current_gap:.4f}',
                        xy=(x[-1], (self.epoch_val_metrics['loss'][-1] + self.epoch_train_metrics['loss'][-1])/2),
                        xytext=(5, 0),
                        textcoords='offset points',
                        ha='left',
                        va='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
                    )
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_learning_curve_analysis(self, ax):
        """Plot learning curve analysis"""
        ax.set_title('Learning Curve Analysis', fontsize=12)
        
        # This is placeholder content - actual implementation would depend on
        # specific learning curve analysis methods relevant to TripoSR
        if self.epoch_train_metrics['loss'] and len(self.epoch_train_metrics['loss']) > 1:
            x = list(range(1, len(self.epoch_train_metrics['loss']) + 1))
            
            # Plot actual learning curve
            ax.plot(
                x, 
                self.epoch_train_metrics['loss'], 
                'b-', 
                label='Training Loss',
                marker='o'
            )
            
            # Simple power-law fit (y = a * x^b) for learning curve
            if len(x) > 3:  # Need enough points for a meaningful fit
                try:
                    from scipy import optimize
                    
                    def power_law(x, a, b):
                        return a * np.power(x, b)
                    
                    params, _ = optimize.curve_fit(
                        power_law, 
                        np.array(x), 
                        np.array(self.epoch_train_metrics['loss']),
                        maxfev=5000
                    )
                    
                    # Generate points for the fitted curve
                    x_fit = np.linspace(1, max(x) * 1.5, 100)
                    y_fit = power_law(x_fit, *params)
                    
                    # Plot the fit
                    ax.plot(
                        x_fit, 
                        y_fit, 
                        'r--', 
                        label=f'Fit: y = {params[0]:.4f} * x^{params[1]:.4f}'
                    )
                    
                    # Predict future performance
                    future_epochs = [max(x) + 5, max(x) + 10]
                    future_losses = power_law(np.array(future_epochs), *params)
                    
                    ax.scatter(
                        future_epochs, 
                        future_losses, 
                        color='green', 
                        marker='*', 
                        s=100, 
                        label='Predicted Future'
                    )
                    
                    for epoch, loss in zip(future_epochs, future_losses):
                        ax.annotate(
                            f'Epoch {epoch}: {loss:.4f}',
                            xy=(epoch, loss),
                            xytext=(5, -5),
                            textcoords='offset points',
                            fontsize=8
                        )
                except:
                    ax.text(
                        0.5, 0.5, 
                        'Learning curve fitting requires\nmore training data', 
                        ha='center', va='center',
                        transform=ax.transAxes
                    )
            else:
                ax.text(
                    0.5, 0.5, 
                    'Learning curve fitting requires\nmore training data', 
                    ha='center', va='center',
                    transform=ax.transAxes
                )
        else:
            ax.text(
                0.5, 0.5, 
                'Learning Curve Analysis\n(Requires more training data)', 
                ha='center', va='center',
                transform=ax.transAxes
            )
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def save_metric_history(self, filepath):
        """
        Save metrics history to a file.
        
        Args:
            filepath: Path to save the metrics history
        """
        import json
        
        # Convert data to JSON-serializable format
        history = {
            'train_metrics': {k: [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in l] 
                              for k, l in self.train_metrics.items()},
            'val_metrics': {k: [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in l] 
                            for k, l in self.val_metrics.items()},
            'epoch_train_metrics': {k: [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in l] 
                                    for k, l in self.epoch_train_metrics.items()},
            'epoch_val_metrics': {k: [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in l] 
                                  for k, l in self.epoch_val_metrics.items()},
            'performance': {
                'batch_times': [float(v) for v in self.batch_times],
                'throughputs': [float(v) for v in self.throughputs],
                'gpu_memory_usage': [float(v) for v in self.gpu_memory_usage]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Metrics history saved to {filepath}")
    
    def load_metric_history(self, filepath):
        """
        Load metrics history from a file.
        
        Args:
            filepath: Path to load the metrics history from
        """
        import json
        
        with open(filepath, 'r') as f:
            history = json.load(f)
        
        self.train_metrics = history['train_metrics']
        self.val_metrics = history['val_metrics']
        self.epoch_train_metrics = history['epoch_train_metrics']
        self.epoch_val_metrics = history['epoch_val_metrics']
        self.batch_times = history['performance']['batch_times']
        self.throughputs = history['performance']['throughputs']
        self.gpu_memory_usage = history['performance']['gpu_memory_usage']
        
        print(f"Metrics history loaded from {filepath}")
        
        # Update the dashboard with loaded metrics
        self.update_dashboard(force=True)


# Example usage:
"""
# Create the metrics visualizer
visualizer = MetricsVisualizer(metric_names=['loss', 'psnr', 'ssim'])
visualizer.initialize_dashboard()

# In training loop:
for epoch in range(epochs):
    # Training loop
    for batch in train_loader:
        # Forward, backward, optimize...
        
        # Update metrics
        metrics = {
            'loss': loss.item(),
            'psnr': psnr_value,
            'ssim': ssim_value,
            'batch_time': batch_time,
            'throughput': samples / batch_time,
            'gpu_memory': gpu_memory_usage
        }
        visualizer.add_train_metrics(metrics, step=step, epoch=epoch)
        
        # Update dashboard periodically
        if step % 10 == 0:
            visualizer.update_dashboard()
    
    # After validation
    val_metrics = {
        'loss': val_loss,
        'psnr': val_psnr,
        'ssim': val_ssim
    }
    visualizer.add_epoch_metrics(train_metrics, val_metrics, epoch)
    visualizer.update_dashboard(force=True)

# Generate final report
visualizer.create_metrics_report(save_path='training_report.png')
"""
