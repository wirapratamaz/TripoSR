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
    
    return mesh


def generate(image, mc_resolution, reference_model=None, formats=["obj", "glb"], 
             model_quality="Standard", texture_quality=7, smoothing_factor=0.3):
    try:
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Optimize quality settings for faster generation
        quality_settings = {
            "Konsep": {"chunk_size": 32768, "detail_factor": 0.5},  # Increased chunk size, lower detail
            "Standar": {"chunk_size": 16384, "detail_factor": 0.7}, # Balanced settings
            "Tinggi": {"chunk_size": 8192, "detail_factor": 1.0}    # Higher detail but slower
        }
        
        # Apply settings based on model quality
        model.renderer.set_chunk_size(quality_settings[model_quality]["chunk_size"])
        
        # Generate scene codes with optimization
        with torch.inference_mode():  # Faster than no_grad()
            scene_codes = model(image, device=device)
            
            # Extract mesh with optimized resolution
            mesh = model.extract_mesh(
                scene_codes, 
                True, 
                resolution=min(mc_resolution, 192)  # Cap resolution for faster processing
            )[0]
        
        # Quick orientation fixes
        mesh = to_gradio_3d_orientation(mesh)
        mesh = fix_model_orientation(mesh)
        
        # Optimize export process
        rv = []
        for format in formats:
            mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
            if format == "glb":
                mesh.export(mesh_path.name, file_type="glb")
            else:
                mesh.export(
                    mesh_path.name,
                    file_type="obj",
                    include_texture=True
                )
            rv.append(mesh_path.name)
        
        rv.extend([0.0, 0.0, 0.0, "Processing complete"])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return rv
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error("GPU memory error. Try 'Konsep' quality or lower resolution.")
        else:
            raise gr.Error(f"Generation error: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_obj, mesh_glb, f1, cd, iou, metrics_text = generate(
        preprocessed, 128, None, ["obj", "glb"],
        "Standar", 7, 0.3
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
                        choices=["Konsep", "Standar", "Tinggi"],
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
                gr.Markdown("Catatan: Unduh untuk mendapatkan pengalaman melihat terbaik.")
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Model Output (Format GLB)",
                    interactive=False,
                    height=600,                         # Larger preview
                )
                gr.Markdown("Catatan: Unduh untuk mendapatkan pengalaman melihat terbaik.")
            with gr.Column():
                with gr.Group():
                    evaluation_box = gr.Textbox(
                        label="Metrik Evaluasi Model",
                        value="Metrik evaluasi akan muncul di sini setelah pembuatan",
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
