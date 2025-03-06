import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
import trimesh
from PIL import Image
from functools import partial

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


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
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
    
    return mesh


def generate(image, mc_resolution, reference_model=None, formats=["obj", "glb"], 
             model_quality="Standard", texture_quality=7, smoothing_factor=0.3):
    try:
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Map model quality to internal settings
        quality_settings = {
            "Draft": {"chunk_size": 16384, "detail_factor": 0.7},
            "Standard": {"chunk_size": 8192, "detail_factor": 1.0},
            "High": {"chunk_size": 4096, "detail_factor": 1.3},
            # Add Indonesian translations
            "Draft": {"chunk_size": 16384, "detail_factor": 0.7},
            "Standar": {"chunk_size": 8192, "detail_factor": 1.0},
            "Tinggi": {"chunk_size": 4096, "detail_factor": 1.3}
        }
        
        # Apply settings based on model quality
        model.renderer.set_chunk_size(quality_settings[model_quality]["chunk_size"])
        detail_factor = quality_settings[model_quality]["detail_factor"]
        
        # Generate scene codes with adjusted parameters
        scene_codes = model(image, device=device)
        
        # Extract mesh with adjusted parameters - removed texture_quality parameter
        mesh = model.extract_mesh(
            scene_codes, 
            True, 
            resolution=mc_resolution
        )[0]
        
        # Apply mesh smoothing if needed
        if smoothing_factor > 0:
            mesh = mesh.smoothed(factor=smoothing_factor)
        
        # Apply standard orientation transformation
        mesh = to_gradio_3d_orientation(mesh)
        
        # Apply additional fixes for better display
        mesh = fix_model_orientation(mesh)
        
        # Load reference model if provided
        ground_truth_mesh = None
        if reference_model is not None:
            try:
                ground_truth_mesh = trimesh.load(reference_model.name)
                logging.info(f"Loaded reference model: {reference_model.name}")
            except Exception as e:
                logging.error(f"Error loading reference model: {str(e)}")
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(mesh, ground_truth_mesh)
        
        # Export meshes with proper format handling
        rv = []
        for format in formats:
            mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
            if format == "glb":
                # For GLB format, ensure proper export settings
                mesh.export(mesh_path.name, file_type="glb")
            else:
                # For OBJ format, include materials and textures
                mesh.export(
                    mesh_path.name,
                    file_type="obj",
                    include_texture=True,
                    write_materials=True
                )
            rv.append(mesh_path.name)
        
        # Format metrics for display
        metrics_text = (
            f"F1-Score: {metrics['f1_score']:.3f}\n"
            f"Chamfer Distance: {metrics['chamfer_distance']:.3f}\n"
            f"IoU Score: {metrics['iou_score']:.3f}\n\n"
            f"Mesh Quality Metrics:\n"
            f"- Vertices: {metrics['vertices']}\n"
            f"- Faces: {metrics['faces']}\n"
            f"- Watertight: {'Yes' if metrics['watertight'] > 0.5 else 'No'}\n"
            f"- Regularity: {metrics['regularity']:.3f}\n"
            f"- Area Uniformity: {metrics['area_uniformity']:.3f}\n"
            f"- Generation Settings: {model_quality} quality, {mc_resolution} resolution"
        )
        
        comparison_note = "\n\nNote: Using estimated metrics (no reference model provided)" if ground_truth_mesh is None else "\n\nNote: Using comparison against reference model"
        metrics_text += comparison_note
        
        # Add metrics to return values
        rv.extend([
            metrics["f1_score"],
            metrics["chamfer_distance"],
            metrics["iou_score"],
            metrics_text
        ])
        
        # Clear CUDA cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return rv
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error("GPU out of memory. Try using a lower resolution or 'Draft' quality setting.")
        else:
            raise gr.Error(f"Error generating model: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_obj, mesh_glb, f1, cd, iou, metrics_text = generate(
        preprocessed, 128, None, ["obj", "glb"],
        "Standard", 7, 0.3  # Default values for the parameters
    )
    return preprocessed, mesh_obj, mesh_glb, f1, cd, iou, metrics_text


with gr.Blocks(title="Generasi Model 3D") as interface:
    gr.Markdown(
        """    
# Generasi Model 3D dari Gambar

Unggah gambar untuk menghasilkan model 3D dengan parameter yang dapat disesuaikan.

## Fine-Tuning Parameters

- **Rasio Latar Depan**: Mengontrol seberapa banyak gambar yang dianggap sebagai latar depan saat pemrosesan. Nilai lebih tinggi akan lebih fokus pada objek utama.
- **Resolusi Marching Cubes**: Mengontrol tingkat detail mesh 3D. Nilai lebih tinggi menciptakan model lebih detail tetapi membutuhkan daya pemrosesan lebih besar.
- **Kualitas Model**: Mengatur tingkat kualitas keseluruhan, mempengaruhi waktu pemrosesan dan detail hasil:
  - Draft: Lebih cepat tapi kurang detail
  - Standar: Pilihan seimbang untuk kebanyakan kasus
  - Tinggi: Lebih detail tapi pemrosesan lebih lambat
- **Kualitas Tekstur**: Mengontrol detail tekstur yang diterapkan pada model. Nilai lebih tinggi menciptakan tekstur lebih detail.
- **Penghalusan Mesh**: Menerapkan penghalusan pada model akhir. Nilai lebih tinggi menciptakan permukaan lebih halus tapi mungkin kehilangan detail halus.

## Tips:
1. Jika hasil tidak memuaskan, coba sesuaikan parameter rasio latar depan dan penghalusan mesh.
2. Untuk model lebih detail, tingkatkan Resolusi Marching Cubes dan atur Kualitas Model ke "Tinggi".
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
                    label="Gambar Input",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Gambar Terproses", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Hapus Latar Belakang", value=True
                    )
                    foreground_ratio = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Rasio Latar Depan",
                    )
                    mc_resolution = gr.Slider(
                        minimum=64,
                        maximum=256,
                        value=128,
                        step=32,
                        label="Resolusi Marching Cubes",
                    )
                    model_quality = gr.Radio(
                        choices=["Draft", "Standar", "Tinggi"],
                        value="Standar",
                        label="Kualitas Model",
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
                        label="Faktor Penghalusan",
                    )
                    reference_model = gr.File(
                        label="Model Referensi (Opsional)",
                        file_types=[".obj", ".glb", ".stl"],
                    )
            with gr.Row():
                submit = gr.Button("Buat Model 3D", variant="primary")
                evaluation_info = gr.Button("📊 Info Evaluasi", elem_id="evaluation_info")
        with gr.Column():
            with gr.Tab("OBJ"):
                output_model_obj = gr.Model3D(
                    label="Model Output (Format OBJ)",
                    interactive=False,
                    height=600,                         # Larger preview
                )
                gr.Markdown("Note: Download to get the best viewing experience.")
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Model Output (Format GLB)",
                    interactive=False,
                    height=600,                         # Larger preview
                )
                gr.Markdown("Note: Download to get the best viewing experience.")
            with gr.Column():
                with gr.Group():
                    evaluation_box = gr.Textbox(
                        label="Model Evaluation Metrics",
                        value="Evaluation metrics will appear here after generation",
                        interactive=False
                    )
                    with gr.Row():
                        f1_score = gr.Number(label="F1-Score", value=0.0, interactive=False)
                        chamfer_dist = gr.Number(label="Chamfer Distance", value=0.0, interactive=False)
                        iou_score = gr.Number(label="IoU Score", value=0.0, interactive=False)
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/garuda-wisnu-kencana.png",
                "examples/tapel-barong1.png",
                "examples/tapel-barong2.png",
                "examples/pintu-belok.png",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj, output_model_glb, f1_score, chamfer_dist, iou_score, evaluation_box],
            cache_examples=False,
            fn=partial(run_example),
            label="Contoh",
            examples_per_page=20,
        )
    
    # Create a popup for evaluation metrics info
    evaluation_info_md = gr.Markdown(visible=False)
    
    def show_evaluation_info():
        with open("evaluation_metrics.md", "r") as f:
            return f.read()
    
    evaluation_info.click(
        fn=show_evaluation_info,
        inputs=[],
        outputs=[evaluation_info_md],
    )
        
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=lambda img, res, ref, qual, tex, smooth: generate(
            img, res, ref, ["obj", "glb"], qual, tex, smooth
        ),
        inputs=[processed_image, mc_resolution, reference_model, model_quality, texture_quality, smoothing_factor],
        outputs=[output_model_obj, output_model_glb, f1_score, chamfer_dist, iou_score, evaluation_box],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None, 
        server_port=args.port
    )
