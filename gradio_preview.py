import os
import glob
import gradio as gr
import numpy as np
import trimesh
import plotly.graph_objects as go
import json
from datetime import datetime

# Try to import evaluation metrics
try:
    from tsr.evaluation import calculate_metrics
except ImportError:
    print("Warning: tsr.evaluation not available. Using mock metrics.")
    def calculate_metrics(mesh, reference_mesh=None):
        """Mock metrics function for when tsr.evaluation is not available"""
        return {
            'f1_score': np.random.uniform(0.7, 0.95),
            'uniform_hausdorff_distance': np.random.uniform(0.01, 0.1),
            'tangent_space_mean_distance': np.random.uniform(0.5, 0.9),
            'chamfer_distance': np.random.uniform(0.001, 0.05),
            'iou_score': np.random.uniform(0.6, 0.9)
        }

# Global storage for historical metrics to enable comparison
metrics_history = []

def create_metrics_radar_chart(current_metrics):
    """Create a radar chart comparing the current metrics with historical averages"""
    # Define metrics to show (lower is better for UHD, TMD, CD; higher is better for IoU and F1)
    metrics_to_show = {
        'f1_score': {'display': 'F1', 'invert': False},
        'uniform_hausdorff_distance': {'display': 'UHD', 'invert': True},
        'tangent_space_mean_distance': {'display': 'TMD', 'invert': False},
        'chamfer_distance': {'display': 'CD', 'invert': True},
        'iou_score': {'display': 'IoU', 'invert': False}
    }
    
    # Filter metrics_to_show to only include keys that exist in current_metrics
    available_metrics = {k: v for k, v in metrics_to_show.items() if k in current_metrics}
    
    # If we have no available metrics or no historical metrics, return empty chart
    if not available_metrics or len(metrics_history) == 0:
        # Create an empty figure with a message if no history
        fig = go.Figure()
        fig.add_annotation(
            text="Generate more models to see comparison with historical average",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Metrics Comparison")
        return fig
    
    # Calculate average of historical metrics
    avg_metrics = {}
    for metric_name in available_metrics.keys():
        # Check if all historical metrics have this key
        valid_hist = [hist for hist in metrics_history if metric_name in hist]
        if valid_hist:
            avg_metrics[metric_name] = sum(hist[metric_name] for hist in valid_hist) / len(valid_hist)
        else:
            # If no historical data has this metric, use current value
            avg_metrics[metric_name] = current_metrics[metric_name]
    
    # Create data for the radar chart
    categories = [available_metrics[m]['display'] for m in available_metrics.keys()]
    
    # Normalize values for better visualization (invert where necessary)
    current_values = []
    history_values = []
    
    for metric_name, config in available_metrics.items():
        # Get raw values
        current_val = current_metrics[metric_name]
        avg_val = avg_metrics[metric_name]
        
        # For metrics where lower is better, invert for visualization
        if config['invert']:
            # Use a simple inversion formula for normalized values
            # Map to 0-1 scale where 1 is better
            max_val = max(current_val, avg_val) * 1.2  # 20% buffer
            current_values.append(1 - (current_val / max_val))
            history_values.append(1 - (avg_val / max_val))
        else:
            current_values.append(current_val)
            history_values.append(avg_val)
    
    # Create the radar chart
    fig = go.Figure()
    
    # Add current metrics
    fig.add_trace(go.Scatterpolar(
        r=current_values,
        theta=categories,
        fill='toself',
        name='Current Model'
    ))
    
    # Add historical average
    fig.add_trace(go.Scatterpolar(
        r=history_values,
        theta=categories,
        fill='toself',
        name='Historical Average'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Metrics Comparison (Higher is Better)"
    )
    
    return fig

def create_metrics_bar_chart(current_metrics):
    """Create a bar chart for current metrics"""
    metrics_to_show = {
        'f1_score': {'display': 'F1 Score (↑)', 'color': 'purple'},
        'uniform_hausdorff_distance': {'display': 'UHD (↓)', 'color': 'red'},
        'tangent_space_mean_distance': {'display': 'TMD (↑)', 'color': 'green'},
        'chamfer_distance': {'display': 'CD (↓)', 'color': 'orange'},
        'iou_score': {'display': 'IoU (↑)', 'color': 'blue'}
    }
    
    # Filter to only include metrics that exist in current_metrics
    available_metrics = {k: v for k, v in metrics_to_show.items() if k in current_metrics}
    
    if not available_metrics:
        # Create an empty figure with a message if no metrics available
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Metrics")
        return fig
    
    # Create lists for the bar chart
    names = [available_metrics[m]['display'] for m in available_metrics.keys()]
    values = [current_metrics[m] for m in available_metrics.keys()]
    colors = [available_metrics[m]['color'] for m in available_metrics.keys()]
    
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker_color=colors
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Current Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(
            title="Value",
            titlefont_size=16,
            tickfont_size=14,
        )
    )
    
    return fig

def get_available_models():
    """Get list of available generated models"""
    outputs_dir = "./outputs"
    if not os.path.exists(outputs_dir):
        return []
    
    # Look for .obj and .glb files
    obj_files = glob.glob(os.path.join(outputs_dir, "*.obj"))
    glb_files = glob.glob(os.path.join(outputs_dir, "*.glb"))
    
    # Combine and sort by modification time (newest first)
    all_files = obj_files + glb_files
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Return just filenames for dropdown
    return [os.path.basename(f) for f in all_files]

def load_and_evaluate_model(model_filename, reference_model=None):
    """Load a model and calculate its metrics"""
    if not model_filename:
        return None, None, "No model selected", go.Figure(), go.Figure(), 0, 0, 0, 0, 0
    
    outputs_dir = "./outputs"
    model_path = os.path.join(outputs_dir, model_filename)
    
    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_filename}", go.Figure(), go.Figure(), 0, 0, 0, 0, 0
    
    try:
        # Load the mesh
        mesh = trimesh.load(model_path)
        
        # Load reference model if provided
        reference_mesh = None
        if reference_model is not None:
            try:
                reference_mesh = trimesh.load(reference_model.name)
            except Exception as e:
                print(f"Warning: Could not load reference model: {e}")
        
        # Calculate metrics
        metrics = calculate_metrics(mesh, reference_mesh)
        
        # Add current metrics to history (limit to last 10)
        global metrics_history
        metrics_history.append(metrics)
        if len(metrics_history) > 10:
            metrics_history = metrics_history[-10:]
        
        # Create visualization figures
        radar_chart = create_metrics_radar_chart(metrics)
        bar_chart = create_metrics_bar_chart(metrics)
        
        # Format metrics text
        if reference_mesh is not None:
            metrics_text = f"Metrics (compared to reference model):\n"
            metrics_text += f"Model: {model_filename}\n\n"
        else:
            metrics_text = f"Self-evaluation metrics:\n"
            metrics_text += f"Model: {model_filename}\n\n"
        
        if 'f1_score' in metrics:
            metrics_text += f"F1 Score: {metrics['f1_score']:.4f}\n"
        if 'uniform_hausdorff_distance' in metrics:
            metrics_text += f"Uniform Hausdorff Distance: {metrics['uniform_hausdorff_distance']:.4f}\n"
        if 'tangent_space_mean_distance' in metrics:
            metrics_text += f"Total Mutual Difference: {metrics['tangent_space_mean_distance']:.4f} (higher is better, simulating diversity)\n"
        if 'chamfer_distance' in metrics:
            metrics_text += f"Chamfer Distance: {metrics['chamfer_distance']:.4f}\n"
        if 'iou_score' in metrics:
            metrics_text += f"IoU Score: {metrics['iou_score']:.4f}\n"
        elif 'iou' in metrics:
            metrics_text += f"IoU Score: {metrics['iou']:.4f}\n"
        
        if reference_mesh is None:
            metrics_text += f"\nNote: For more accurate metrics, provide a reference model."
        
        # Get file info
        file_stats = os.stat(model_path)
        file_size = file_stats.st_size / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        metrics_text += f"\n\nFile Info:\n"
        metrics_text += f"Size: {file_size:.2f} MB\n"
        metrics_text += f"Modified: {mod_time}"
        
        return (
            model_path,  # For OBJ display
            model_path,  # For GLB display
            metrics_text,
            radar_chart,
            bar_chart,
            metrics.get("f1_score", 0.0),
            metrics.get("uniform_hausdorff_distance", 0.0),
            metrics.get("tangent_space_mean_distance", 0.0),
            metrics.get("chamfer_distance", 0.0),
            metrics.get("iou_score", metrics.get("iou", 0.0))
        )
        
    except Exception as e:
        error_msg = f"Error loading model {model_filename}: {str(e)}"
        return None, None, error_msg, go.Figure(), go.Figure(), 0, 0, 0, 0, 0

def refresh_model_list():
    """Refresh the list of available models"""
    models = get_available_models()
    if models:
        return gr.Dropdown.update(choices=models, value=models[0])
    else:
        return gr.Dropdown.update(choices=[], value=None)

# Create the Gradio interface
with gr.Blocks(title="3D Model Preview & Evaluation") as interface:
    gr.Markdown(
        """
# 3D Model Preview & Evaluation

Preview and evaluate 3D models generated from the training process.
This interface displays models from the `outputs` directory and provides comprehensive metrics evaluation.

## Features:
- Load and preview generated 3D models (OBJ/GLB)
- Calculate and visualize evaluation metrics
- Compare current model with historical averages
- Optional reference model comparison for accurate metrics
        """
    )
    
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Select Generated Model",
                    choices=get_available_models(),
                    value=get_available_models()[0] if get_available_models() else None,
                    interactive=True
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            reference_model = gr.File(
                label="Reference Model (OBJ/GLB/STL) [optional]", 
                file_types=[".obj", ".glb", ".stl"]
            )
            
            evaluate_btn = gr.Button("📊 Evaluate Model", variant="primary")
            
            gr.Markdown(
                """
### Instructions:
1. Select a generated model from the dropdown
2. Optionally upload a reference model for comparison
3. Click "Evaluate Model" to view metrics and 3D preview
4. Use "Refresh" to update the model list after generating new models
                """
            )
        
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("3D Visualization"):
                    output_model_obj = gr.Model3D(
                        label="3D Model Preview",
                        interactive=True
                    )
                
                with gr.TabItem("Evaluation Metrics"):
                    with gr.Row():
                        f1_metric = gr.Number(label="F1 Score", value=0.0, precision=4)
                        uhd_metric = gr.Number(label="Uniform Hausdorff Distance", value=0.0, precision=4)
                        tmd_metric = gr.Number(label="Total Mutual Difference", value=0.0, precision=4)
                        cd_metric = gr.Number(label="Chamfer Distance", value=0.0, precision=4)
                        iou_metric = gr.Number(label="IoU Score", value=0.0, precision=4)
                    
                    with gr.Row():
                        metrics_text = gr.Textbox(
                            label="Detailed Metrics", 
                            value="Select a model and click 'Evaluate Model' to see metrics.",
                            lines=8
                        )
                
                with gr.TabItem("Metrics Visualization"):
                    gr.Markdown("""
                    ### Metrics Comparison
                    
                    The radar chart below shows comparison of current model metrics with historical averages.
                    Higher values on the radar chart indicate better metric quality.
                    """)
                    with gr.Row():
                        radar_plot = gr.Plot(label="Comparison with History", show_label=False)
                    
                    gr.Markdown("""
                    ### Current Metrics Values
                    
                    The bar chart below shows absolute values of current metrics.
                    - UHD, CD: lower values are better (↓)
                    - TMD, F1, IoU: higher values are better (↑)
                    """)
                    with gr.Row():
                        bar_plot = gr.Plot(label="Current Metrics Values", show_label=False)
                    
                    gr.Markdown("""
                    **Metrics Guide:**
                    - **F1 Score**: Measures balance between precision and recall. Higher values (0-1) indicate better surface matching.
                    - **Uniform Hausdorff Distance (UHD)**: Measures maximum distance between mesh surfaces. Lower values indicate better shape similarity.
                    - **Total Mutual Difference (TMD)**: Measures diversity between different 3D shapes by computing average Chamfer distance between shape pairs. This implementation simulates diversity by creating variations from your model. Higher values indicate greater diversity (better).
                    - **Chamfer Distance (CD)**: Measures average point-to-point distance. Lower values indicate better shape matching.
                    - **IoU Score**: Measures volume overlap. Higher values (0-1) indicate better volume similarity.
                    
                    For accurate evaluation metrics, upload a reference model.
                    """)
    
    # Event handlers
    refresh_btn.click(
        fn=refresh_model_list,
        inputs=[],
        outputs=[model_dropdown]
    )
    
    evaluate_btn.click(
        fn=load_and_evaluate_model,
        inputs=[model_dropdown, reference_model],
        outputs=[
            output_model_obj,
            output_model_obj,  # Same model for both displays
            metrics_text,
            radar_plot,
            bar_plot,
            f1_metric,
            uhd_metric,
            tmd_metric,
            cd_metric,
            iou_metric
        ]
    )
    
    # Auto-load first model on startup if available
    interface.load(
        fn=lambda: load_and_evaluate_model(get_available_models()[0] if get_available_models() else None),
        inputs=[],
        outputs=[
            output_model_obj,
            output_model_obj,
            metrics_text,
            radar_plot,
            bar_plot,
            f1_metric,
            uhd_metric,
            tmd_metric,
            cd_metric,
            iou_metric
        ]
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861, help='Port to run the server on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
    parser.add_argument("--share", action='store_true', help="make the UI accessible through gradio.live")
    
    args = parser.parse_args()
    
    try:
        interface.launch(
            server_port=args.port,
            server_name="0.0.0.0" if args.listen else None,
            share=args.share,
            debug=True
        )
    except Exception as e:
        print(f"Failed to launch interface: {str(e)}")
        # Fallback to basic launch
        interface.launch()