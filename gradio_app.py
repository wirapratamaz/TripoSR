import logging
import os
import tempfile
import time

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s:%(message)s'
)

import gradio as gr
import numpy as np
import rembg
import torch
import trimesh
from PIL import Image
from functools import partial
import plotly.graph_objects as go
import plotly.express as px
import json
import io

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from tsr.evaluation import calculate_metrics

import argparse


# Configure CUDA memory settings
if torch.cuda.is_available():
    device = "cuda:0"
    # Lower default chunk size to reduce memory usage
    default_chunk_size = 8192
else:
    device = "cpu"
    default_chunk_size = 8192

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(default_chunk_size)
model.to(device)

rembg_session = rembg.new_session()

# Global storage for historical metrics to enable comparison
metrics_history = []

# Clear metrics history to avoid issues with key name changes
def reset_metrics_history():
    global metrics_history
    metrics_history = []

reset_metrics_history()  # Reset at startup

def create_metrics_radar_chart(current_metrics):
    """Create a radar chart comparing the current metrics with historical averages"""
    # Define metrics to show (lower is better for UHD, TMD, CD; higher is better for IoU and F1)
    metrics_to_show = {
        'f1_score': {'display': 'F1', 'invert': False},
        'uniform_hausdorff_distance': {'display': 'UHD', 'invert': True},
        'tangent_space_mean_distance': {'display': 'TMD', 'invert': True},
        'chamfer_distance': {'display': 'CD', 'invert': True},
        'iou': {'display': 'IoU', 'invert': False}
    }
    
    # If we have historical metrics, calculate average
    if len(metrics_history) > 0:
        # Calculate average of historical metrics
        avg_metrics = {}
        for metric_name in metrics_to_show.keys():
            try:
                avg_metrics[metric_name] = sum(
                    hist.get(metric_name, 
                            hist.get('iou_score' if metric_name == 'iou' else metric_name, 0.0)) 
                    for hist in metrics_history
                ) / len(metrics_history)
            except Exception:
                avg_metrics[metric_name] = 0.0
        
        # Create data for the radar chart
        categories = [metrics_to_show[m]['display'] for m in metrics_to_show.keys()]
        
        # Normalize values for better visualization (invert where necessary)
        current_values = []
        history_values = []
        
        for metric_name, config in metrics_to_show.items():
            try:
                # Get raw values
                current_val = current_metrics[metric_name]
                avg_val = avg_metrics[metric_name]
                
                # For metrics where lower is better, invert for visualization
                if config['invert']:
                    # Use a simple inversion formula for normalized values
                    # Map to 0-1 scale where 1 is better
                    # Avoid division by zero
                    max_val = max(current_val, avg_val) * 1.2  # 20% buffer
                    
                    if max_val == 0:
                        # Both values are zero, display as perfect score
                        current_values.append(1.0)
                        history_values.append(1.0)
                    else:
                        current_values.append(1 - (current_val / max_val))
                        history_values.append(1 - (avg_val / max_val))
                else:
                    # Scale the values to a reasonable range
                    scale_factor = max(max(current_val, avg_val) * 1.2, 0.001)  # Avoid zero
                    current_values.append(current_val / scale_factor)
                    history_values.append(avg_val / scale_factor)
            except Exception:
                # Use default values when error occurs
                current_values.append(0.5)
                history_values.append(0.5)
        
        # Create the radar chart
        try:
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
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True
            )
            
            return fig
        except Exception:
            # Return empty figure on error
            return go.Figure()
    else:
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

def create_metrics_bar_chart(current_metrics):
    """Create a bar chart for current metrics"""
    metrics_to_show = {
        'f1_score': {'display': 'F1 Score (↑)', 'color': 'purple'},
        'uniform_hausdorff_distance': {'display': 'UHD (↓)', 'color': 'red'},
        'tangent_space_mean_distance': {'display': 'TMD (↓)', 'color': 'orange'},
        'chamfer_distance': {'display': 'CD (↓)', 'color': 'green'},
        'iou': {'display': 'IoU (↑)', 'color': 'blue'}
    }
    
    # Prepare data for bar chart
    labels = [metrics_to_show[m]['display'] for m in metrics_to_show.keys()]
    
    # Get values with fallback for missing keys (for backward compatibility)
    values = []
    for m in metrics_to_show.keys():
        try:
            # Check if the key exists in current_metrics, with fallback
            if m in current_metrics:
                values.append(current_metrics[m])
            elif m == 'iou' and 'iou_score' in current_metrics:
                # Handle possible old format
                values.append(current_metrics['iou_score'])
            else:
                values.append(0.0)  # Default value if missing
        except Exception:
            values.append(0.0)  # Default on error
    
    colors = [metrics_to_show[m]['color'] for m in metrics_to_show.keys()]
    
    # Create the bar chart
    try:
        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=colors)
        ])
        
        fig.update_layout(
            title="Current Metrics Values",
            xaxis_title="Metrics",
            yaxis_title="Value"
        )
        
        return fig
    except Exception:
        # Return empty figure on error
        return go.Figure()

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        return Image.fromarray((image * 255.0).astype(np.uint8))

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
            
    # Ensure image size is reasonable
    max_size = 512
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
    return image


def fix_model_orientation(mesh):
    """Fix the orientation of the model for proper display"""
    # Rotate 90 degrees around X axis to match standard orientation
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0],
        point=[0, 0, 0]
    )
    mesh.apply_transform(rotation_matrix)
    
    # Center the mesh
    mesh.vertices -= mesh.vertices.mean(axis=0)
    
    # Scale to fit in a unit cube
    scale = 1.0 / max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices *= scale
    
    # Fix normals to ensure proper rendering
    mesh.fix_normals()
    
    # Ensure material properties aren't too reflective
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
        # Reduce specularity to minimize bright reflections
        if hasattr(mesh.visual.material, 'specular'):
            mesh.visual.material.specular = [0.1, 0.1, 0.1, 1.0]
        # Set ambient color to ensure better visibility
        if hasattr(mesh.visual.material, 'ambient'):
            mesh.visual.material.ambient = [0.6, 0.6, 0.6, 1.0]
        # Adjust shininess to reduce glossy appearance
        if hasattr(mesh.visual.material, 'shininess'):
            mesh.visual.material.shininess = 0.1

    return mesh


def generate(image, mc_resolution, reference_model=None, formats=["obj", "glb"], 
             model_quality="Standard", texture_quality=7, smoothing_factor=0.3):
    try:
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Create a permanent output directory
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Fixed quality settings to match dropdown options
        quality_settings = {
            "Draft": {"chunk_size": 32768, "detail_factor": 0.5},
            "Standard": {"chunk_size": 16384, "detail_factor": 0.7},
            "High": {"chunk_size": 8192, "detail_factor": 1.0}
        }
        
        # Set chunk size based on quality setting
        if model_quality in quality_settings:
            model.renderer.set_chunk_size(quality_settings[model_quality]["chunk_size"])
        else:
            # Default to Standard if quality setting not found
            model.renderer.set_chunk_size(quality_settings["Standard"]["chunk_size"])
        
        with torch.inference_mode():
            scene_codes = model(image, device=device)
            mesh = model.extract_mesh(
                scene_codes, 
                True, 
                resolution=min(mc_resolution, 192)
            )[0]
        
        mesh = to_gradio_3d_orientation(mesh)
        mesh = fix_model_orientation(mesh)
        
        # Apply smoothing if requested - using the proper method
        if smoothing_factor > 0:
            # Using laplacian_smooth instead of smoothed with proper parameters
            # This is the correct way to apply smoothing in trimesh
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                from trimesh import smoothing
                # Apply laplacian smoothing with the specified factor as iterations
                iterations = max(1, int(smoothing_factor * 10))  # Convert factor to iterations (1-10)
                smoothing.filter_laplacian(mesh, iterations=iterations)
        
        # Improve texture appearance by normalizing colors
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            # Get vertex colors
            colors = mesh.visual.vertex_colors
            
            # Normalize brightness to prevent extreme bright spots
            # Convert to HSV for better manipulation
            import colorsys
            normalized_colors = np.zeros_like(colors)
            
            for i in range(len(colors)):
                r, g, b = colors[i][0]/255.0, colors[i][1]/255.0, colors[i][2]/255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                
                # Cap brightness (v) to prevent overly bright spots
                v = min(v, 0.95)
                
                # Increase saturation slightly for better visual appeal
                s = min(s * 1.1, 1.0)
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                normalized_colors[i][0] = int(r * 255)
                normalized_colors[i][1] = int(g * 255)
                normalized_colors[i][2] = int(b * 255)
                normalized_colors[i][3] = colors[i][3]  # Keep alpha channel
            
            mesh.visual.vertex_colors = normalized_colors
        
        # Load reference model if provided
        reference_mesh = None
        if reference_model is not None:
            reference_mesh = trimesh.load(reference_model.name)
        
        # Calculate actual metrics
        try:
            metrics = calculate_metrics(mesh, reference_mesh)
        except Exception as metric_error:
            logging.warning(f"Error calculating metrics: {str(metric_error)}")
            # Provide default metrics when calculation fails
            metrics = {
                "f1_score": 0.01,
                "uniform_hausdorff_distance": 0.01,
                "tangent_space_mean_distance": 0.01,
                "chamfer_distance": 0.01,
                "iou": 0.01
            }
        
        # Add current metrics to history (limit to last 10)
        global metrics_history
        metrics_history.append(metrics)
        if len(metrics_history) > 10:
            metrics_history = metrics_history[-10:]
        
        # Create visualization figures
        try:
            radar_chart = create_metrics_radar_chart(metrics)
        except Exception:
            # Create a simple empty chart on error
            radar_chart = go.Figure()
            radar_chart.add_annotation(
                text="Error creating radar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        
        try:
            bar_chart = create_metrics_bar_chart(metrics)
        except Exception:
            # Create a simple empty chart on error
            bar_chart = go.Figure()
            bar_chart.add_annotation(
                text="Error creating bar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        
        # Format metrics text
        try:
            if reference_mesh is not None:
                metrics_text = (
                    f"Metrics (compared to reference model):\n"
                    f"F1 Score: {metrics['f1_score']:.4f}\n"
                    f"Uniform Hausdorff Distance: {metrics['uniform_hausdorff_distance']:.4f}\n"
                    f"Tangent-Space Mean Distance: {metrics['tangent_space_mean_distance']:.4f}\n"
                    f"Chamfer Distance: {metrics['chamfer_distance']:.4f}\n"
                    f"IoU Score: {metrics['iou']:.4f}"
                )
            else:
                metrics_text = (
                    f"Self-evaluation metrics:\n"
                    f"F1 Score: {metrics['f1_score']:.4f}\n"
                    f"Uniform Hausdorff Distance: {metrics['uniform_hausdorff_distance']:.4f}\n"
                    f"Tangent-Space Mean Distance: {metrics['tangent_space_mean_distance']:.4f}\n"
                    f"Chamfer Distance: {metrics['chamfer_distance']:.4f}\n"
                    f"IoU Score: {metrics['iou']:.4f}\n"
                    f"Note: For more accurate metrics, provide a reference model."
                )
        except Exception:
            metrics_text = "Error generating metrics text."
        
        # Save files with permanent paths
        rv = []
        for format in formats:
            try:
                file_path = os.path.join(output_dir, f"model_{timestamp}.{format}")
                # Make sure to use absolute path
                abs_file_path = os.path.abspath(file_path)
                
                if format == "glb":
                    mesh.export(abs_file_path, file_type="glb")
                else:
                    # For OBJ, use improved texture settings
                    mesh.export(
                        abs_file_path,
                        file_type="obj",
                        include_texture=True,
                        include_normals=True,  # Ensure normals are included for better rendering
                        resolver=None,
                        mtl_name=f"model_{timestamp}.mtl"
                    )
                # Log the exported file path for debugging
                logging.info(f"Successfully exported 3D model to: {abs_file_path}")
                rv.append(abs_file_path)
            except Exception as e:
                logging.error(f"Error exporting {format} model: {str(e)}")
                # Add None to maintain the expected return structure
                rv.append(None)
        
        # Add metrics to return values
        rv.extend([
            metrics["f1_score"],
            metrics["uniform_hausdorff_distance"],
            metrics["tangent_space_mean_distance"],
            metrics["chamfer_distance"],
            metrics["iou"],
            metrics_text,
            radar_chart,
            bar_chart
        ])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Return only the first file path for each format and the metrics
        # This ensures we return exactly the expected number of values
        obj_path = None
        glb_path = None
        
        for path in rv:
            if isinstance(path, str):
                if path.endswith('.obj') and obj_path is None:
                    obj_path = path
                elif path.endswith('.glb') and glb_path is None:
                    glb_path = path
        
        # Return exactly the expected values in the expected order
        return [
            obj_path,  # For output_model_obj
            glb_path,  # For output_model_glb
            float(metrics["f1_score"]),  # Ensure numeric type
            float(metrics["uniform_hausdorff_distance"]),  # Ensure numeric type
            float(metrics["tangent_space_mean_distance"]),  # Ensure numeric type
            float(metrics["chamfer_distance"]),  # Ensure numeric type
            float(metrics["iou"]),  # Ensure numeric type
            metrics_text,
            radar_chart,
            bar_chart
        ]
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error("GPU memory error. Try 'Standard' quality or lower resolution.")
        else:
            raise gr.Error(f"Generation error: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    result = generate(
        preprocessed, 128, None, ["obj", "glb"],
        "Standard", 7, 0.3
    )
    # Unpack the result and return all expected values
    return [preprocessed] + result


with gr.Blocks(title="3D Model Generation") as interface:
    gr.Markdown(
        """    
# Generasi Model 3D dari Gambar

Unggah gambar untuk menghasilkan model 3D dengan parameter yang dapat disesuaikan.

## Fine-Tuning Parameters

- **Foreground Ratio**: Mengontrol seberapa banyak gambar yang dianggap sebagai latar depan saat pemrosesan. Nilai lebih tinggi akan lebih fokus pada objek utama.
- **Marching Cubes Resolution**: Mengontrol tingkat detail mesh 3D. Nilai lebih tinggi menciptakan model lebih detail tetapi membutuhkan daya pemrosesan lebih besar.
- **Kualitas Model**: Mengatur tingkat kualitas keseluruhan, mempengaruhi waktu pemrosesan dan detail hasil:
  - Draft: Lebih cepat tapi kurang detail
  - Standar: Pilihan seimbang untuk kebanyakan kasus
  - Tinggi: Lebih detail tapi pemrosesan lebih lambat
- **Kualitas Tekstur**: Mengontrol detail tekstur yang diterapkan pada model. Nilai lebih tinggi menciptakan tekstur lebih detail.
- **Mesh Smoothing**: Menerapkan penghalusan pada model akhir. Nilai lebih tinggi menciptakan permukaan lebih halus tapi mungkin kehilangan detail halus.

## Tips:
1. Jika hasil tidak memuaskan, coba sesuaikan parameter Foreground Ratio dan Mesh Smoothing.
2. Untuk model lebih detail, tingkatkan Marching Cubes Resolution dan atur Kualitas Model ke "Tinggi".
3. Lebih baik nonaktifkan "Hapus Latar Belakang" untuk contoh yang disediakan (kecuali yang terakhir) karena sudah diproses sebelumnya.
4. Nonaktifkan opsi "Hapus Latar Belakang" hanya jika gambar input Anda adalah RGBA dengan latar belakang transparan, konten gambar terpusat dan menempati lebih dari 70% lebar atau tinggi gambar.
5. Untuk metrik evaluasi yang akurat, unggah model referensi dalam format OBJ, GLB atau STL.
6. Waktu pemrosesan meningkat dengan pengaturan resolusi dan kualitas yang lebih tinggi.
    """
    )
    
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Unggah Gambar Disini",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Foreground Ratio",
                    )
                    mc_resolution = gr.Slider(
                        minimum=64,
                        maximum=256,
                        value=128,
                        step=16,
                        label="Marching Cubes Resolution",
                    )
                    model_quality = gr.Dropdown(
                        choices=["Draft", "Standard", "High"],
                        value="Standard",
                        label="Model Quality",
                    )
                    texture_quality = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=7,
                        step=1,
                        label="Kualitas Tekstur",
                    )
                    smoothing_factor = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Mesh Smoothing",
                    )
                    reference_model = gr.File(
                        label="Reference Model (OBJ/GLB/STL) [optional]", 
                        file_types=[".obj", ".glb", ".stl"]
                    )
                    submit = gr.Button("Generate 3D Model", variant="primary")
                    evaluation_info = gr.Button("ℹ️ Metric Information", size="sm")
                
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("3D Visualisasi"):
                        output_model_obj = gr.Model3D(
                            label="Model 3D (OBJ)",
                            interactive=True,
                            clear_color=[0.0, 0.0, 0.0, 0.0],  # Transparent background
                            camera_position=[0, 0, 2.0],  # Set default camera position
                            camera_target=[0, 0, 0],  # Look at the center
                            height=600  # Increased height for better visibility
                        )
                        output_model_glb = gr.Model3D(
                            label="Model 3D (GLB)",
                            interactive=True,
                            clear_color=[0.0, 0.0, 0.0, 0.0],  # Transparent background
                            camera_position=[0, 0, 2.0],  # Set default camera position
                            camera_target=[0, 0, 0],  # Look at the center
                            height=600  # Increased height for better visibility
                        )
                    with gr.TabItem("Metrik Evaluasi"):
                        with gr.Row():
                            f1_metric = gr.Number(label="F1 Score", value=0.0, precision=4)
                            uhd_metric = gr.Number(label="Uniform Hausdorff Distance", value=0.0, precision=4)
                            tmd_metric = gr.Number(label="Tangent-Space Mean Distance", value=0.0, precision=4)
                            cd_metric = gr.Number(label="Chamfer Distance", value=0.0, precision=4)
                            iou_metric = gr.Number(label="IoU Score", value=0.0, precision=4)
                        
                        with gr.Row():
                            metrics_text = gr.Textbox(
                                label="Metrik Lengkap", 
                                value="Hasilkan model untuk melihat metrik evaluasi.\n\nUntuk perbandingan yang lebih akurat, unggah model referensi.",
                                lines=6
                            )
                    
                    with gr.TabItem("Visualisasi Metrik"):
                        gr.Markdown("""
                        ### Visualisasi Perbandingan Metrik
                        
                        Diagram di bawah menunjukkan perbandingan metrik model saat ini dengan rata-rata historis.
                        Semakin besar nilai pada diagram radar, semakin baik kualitas metrik tersebut.
                        """)
                        with gr.Row():
                            radar_plot = gr.Plot(label="Perbandingan dengan Riwayat", show_label=False)
                        
                        gr.Markdown("""
                        ### Nilai Metrik Saat Ini
                        
                        Diagram batang di bawah menunjukkan nilai absolut dari metrik saat ini.
                        UHD, TMD, CD: nilai lebih rendah lebih baik (↓)
                        IoU: nilai lebih tinggi lebih baik (↑)
                        """)
                        with gr.Row():
                            bar_plot = gr.Plot(label="Nilai Metrik Saat Ini", show_label=False)
                        
                        gr.Markdown("""
                        **Petunjuk Metrik:**
                        - **F1 Score**: Mengukur keseimbangan antara presisi dan recall. Nilai lebih tinggi (0-1) menunjukkan kecocokan permukaan yang lebih baik.
                        - **Uniform Hausdorff Distance (UHD)**: Mengukur jarak maksimum antara permukaan mesh. Nilai lebih rendah menunjukkan kesamaan bentuk yang lebih baik.
                        - **Tangent-Space Mean Distance (TMD)**: Mengukur jarak rata-rata pada ruang tangensial. Nilai lebih rendah menunjukkan kesamaan bentuk lokal yang lebih baik.
                        - **Chamfer Distance (CD)**: Mengukur jarak rata-rata antar titik. Nilai lebih rendah menunjukkan kecocokan bentuk yang lebih baik.
                        - **IoU Score**: Mengukur volume tumpang tindih. Nilai lebih tinggi (0-1) menunjukkan kesamaan volume yang lebih baik.
                        
                        Untuk metrik evaluasi yang akurat, unggah model referensi.
                        """)
        
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/garuda-wisnu-kencana.png",
                "examples/tapel-barong1.png",
                "examples/tapel-barong2.png",
                "examples/pintu-belok.png",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj, output_model_glb, f1_metric, uhd_metric, tmd_metric, cd_metric, iou_metric, metrics_text, radar_plot, bar_plot],
            cache_examples=False,
            fn=partial(run_example),
            label="Contoh",
            examples_per_page=20,
        )
    
    # Create a popup for evaluation metrics info
    evaluation_info_md = gr.Markdown(visible=False, value="""
    ## Evaluation Metrics Information
    
    - **F1 Score**: Measures the balance between precision and recall. Higher values (0-1) indicate better surface matching.
    - **Uniform Hausdorff Distance (UHD)**: Measures the maximum distance between mesh surfaces. Lower values indicate better shape similarity.
    - **Tangent-Space Mean Distance (TMD)**: Measures the average distance in tangential space. Lower values indicate better local shape similarity.
    - **Chamfer Distance (CD)**: Measures the average distance between points. Lower values indicate better shape matching.
    - **IoU Score**: Measures volume overlap. Higher values (0-1) indicate better volume similarity.
    
    For accurate evaluation metrics, upload a reference model.
    """)
    
    def show_evaluation_info():
        return {"visible": True}
    
    evaluation_info.click(
        fn=show_evaluation_info,
        inputs=[],
        outputs=[evaluation_info_md],
    )
        
    submit.click(
        fn=check_input_image,
        inputs=[input_image],
        outputs=[],
    ).then(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).then(
        fn=generate,
        inputs=[
            processed_image,
            mc_resolution,
            reference_model,
            model_quality,
            texture_quality,
            smoothing_factor,
        ],
        outputs=[
            output_model_obj,
            output_model_glb,
            f1_metric,
            uhd_metric,
            tmd_metric,
            cd_metric,
            iou_metric,
            metrics_text,
            radar_plot,
            bar_plot,
        ],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
    parser.add_argument("--share", action='store_true', help="make the UI accessible through gradio.live")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    
    args = parser.parse_args()
    
    # Configure queue before launch
    interface.queue(max_size=args.queuesize)
    
    # Prepare auth tuple
    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)
    
    # Launch with simplified parameters
    try:
        interface.launch(
            server_port=args.port,
            server_name="0.0.0.0" if args.listen else None,
            share=args.share,
            auth=auth,
            debug=True  # Add debug mode to see more detailed errors
        )
    except Exception as e:
        print(f"Failed to launch interface: {str(e)}")
        # Fallback to basic launch if custom configuration fails
        interface.launch()
