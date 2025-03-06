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


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
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


def generate(image, mc_resolution, reference_model=None, formats=["obj", "glb"], 
             model_quality="Standard", texture_quality=7, smoothing_factor=0.3):
    # Map model quality to internal settings
    quality_settings = {
        "Draft": {"chunk_size": 16384, "detail_factor": 0.7},
        "Standard": {"chunk_size": 8192, "detail_factor": 1.0},
        "High": {"chunk_size": 4096, "detail_factor": 1.3}
    }
    
    # Apply settings based on model quality
    model.renderer.set_chunk_size(quality_settings[model_quality]["chunk_size"])
    detail_factor = quality_settings[model_quality]["detail_factor"]
    
    # Generate scene codes with adjusted parameters
    scene_codes = model(image, device=device)
    
    # Extract mesh with adjusted parameters
    mesh = model.extract_mesh(
        scene_codes, 
        True, 
        resolution=mc_resolution,
        texture_quality=texture_quality/10.0  # Normalize to 0.1-1.0 range
    )[0]
    
    # Apply mesh smoothing if needed
    if smoothing_factor > 0:
        mesh = mesh.smoothed(factor=smoothing_factor)
    
    mesh = to_gradio_3d_orientation(mesh)
    
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
    
    # Export meshes
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
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
    
    return rv


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_obj, mesh_glb, f1, cd, iou, metrics_text = generate(
        preprocessed, 256, None, ["obj", "glb"], 
        "Standard", 7, 0.3  # Default values for the new parameters
    )
    return preprocessed, mesh_obj, mesh_glb, f1, cd, iou, metrics_text


with gr.Blocks(title="3D Model Generation") as interface:
    gr.Markdown(
        """    
# 3D Model Generation from Images

Upload an image to generate a 3D model with customizable parameters.

## Fine-Tuning Parameters

- **Foreground Ratio**: Controls how much of the image is considered foreground when processing. Higher values focus more on the central object.
- **Marching Cubes Resolution**: Controls the detail level of the 3D mesh. Higher values create more detailed models but require more processing power.
- **Model Quality**: Sets overall quality level, affecting processing time and result detail:
  - Draft: Faster but less detailed
  - Standard: Balanced option for most cases
  - High: More detailed but slower processing
- **Texture Quality**: Controls the detail of textures applied to the model. Higher values create more detailed textures.
- **Mesh Smoothing**: Applies smoothing to the final model. Higher values create smoother surfaces but may lose fine details.

## Tips:
1. If you find the result is unsatisfied, try adjusting the foreground ratio and mesh smoothing parameters.
2. For more detailed models, increase the Marching Cubes Resolution and set Model Quality to "High".
3. It's better to disable "Remove Background" for the provided examples (except for the last one) since they have been already preprocessed.
4. Otherwise, please disable "Remove Background" option only if your input image is RGBA with transparent background, image contents are centered and occupy more than 70% of image width or height.
5. For accurate evaluation metrics, upload a reference model in OBJ, GLB or STL format.
6. Processing time increases with higher resolution and quality settings.
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
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
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=512,
                        value=256,
                        step=32,
                        info="Higher resolution creates more detailed models but uses more memory"
                    )
                    model_quality = gr.Radio(
                        label="Model Quality",
                        choices=["Draft", "Standard", "High"],
                        value="Standard",
                        info="Higher quality takes longer but produces better results"
                    )
                    texture_quality = gr.Slider(
                        label="Texture Quality",
                        minimum=1,
                        maximum=10,
                        value=7,
                        step=1,
                        info="Higher values produce more detailed textures"
                    )
                    smoothing_factor = gr.Slider(
                        label="Mesh Smoothing",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        info="Higher values produce smoother meshes but may lose detail"
                    )
                    reference_model = gr.File(
                        label="Reference Model (Optional)",
                        file_types=[".obj", ".glb", ".stl"],
                        type="file"
                    )
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
                evaluation_info = gr.Button("📊 Evaluation Info", elem_id="evaluation_info")
        with gr.Column():
            with gr.Tab("OBJ"):
                output_model_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    interactive=False,
                )
                gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Output Model (GLB Format)",
                    interactive=False,
                )
                gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
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
            label="Examples",
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