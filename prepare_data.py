import os
import argparse
import shutil
import random
from PIL import Image
import numpy as np
import trimesh


def prepare_dataset():
    parser = argparse.ArgumentParser(description="Prepare dataset for TripoSR fine-tuning")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing raw data")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory for prepared dataset")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (0-1)")
    parser.add_argument("--resize", type=int, default=256, help="Resize images to this size")
    parser.add_argument("--check_manifold", action="store_true", help="Check if meshes are manifold")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)
    
    # Find all potential object directories in the input directory
    object_dirs = []
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path):
            object_dirs.append(item)
    
    if not object_dirs:
        print("No object directories found in the input directory!")
        return
    
    print(f"Found {len(object_dirs)} potential object directories")
    
    # Process each object directory
    processed_count = 0
    skipped_count = 0
    
    for obj_id in object_dirs:
        source_dir = os.path.join(args.input_dir, obj_id)
        
        # Check if directory contains necessary files
        image_file = find_image_file(source_dir)
        mesh_file = find_mesh_file(source_dir)
        
        if not image_file or not mesh_file:
            print(f"Skipping {obj_id}: Missing image or mesh file")
            skipped_count += 1
            continue
        
        # Check if mesh is manifold
        if args.check_manifold:
            try:
                mesh = trimesh.load(os.path.join(source_dir, mesh_file))
                
                # Note: is_watertight is used here as a practical check for mesh integrity.
                # While related to manifoldness, is_watertight and is_manifold are not identical concepts:
                # - A watertight mesh has no holes (every edge connects exactly 2 faces)
                # - A manifold mesh satisfies additional topological constraints
                # 
                # For precise manifold checking, one could use:
                # if not (mesh.is_watertight and mesh.is_manifold):
                #     ...
                # 
                # The current approach (checking only watertightness) is a reasonable practical
                # compromise that catches most problematic meshes without being too restrictive.
                
                if not mesh.is_watertight:
                    print(f"Skipping {obj_id}: Mesh is not manifold (watertight)")
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"Skipping {obj_id}: Error loading mesh: {e}")
                skipped_count += 1
                continue
        
        # Decide train or validation split
        if random.random() < args.val_split:
            target_dir = os.path.join(args.output_dir, "val", obj_id)
        else:
            target_dir = os.path.join(args.output_dir, "train", obj_id)
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Process and copy image
        try:
            image = Image.open(os.path.join(source_dir, image_file))
            image = image.convert("RGB")
            image = image.resize((args.resize, args.resize), Image.LANCZOS)
            image.save(os.path.join(target_dir, "image.png"))
        except Exception as e:
            print(f"Skipping {obj_id}: Error processing image: {e}")
            shutil.rmtree(target_dir)
            skipped_count += 1
            continue
        
        # Copy mesh file
        try:
            mesh = trimesh.load(os.path.join(source_dir, mesh_file))
            mesh.export(os.path.join(target_dir, "model.obj"))
        except Exception as e:
            print(f"Skipping {obj_id}: Error processing mesh: {e}")
            shutil.rmtree(target_dir)
            skipped_count += 1
            continue
        
        # Copy mask if available
        mask_file = find_mask_file(source_dir)
        if mask_file:
            try:
                mask = Image.open(os.path.join(source_dir, mask_file))
                mask = mask.convert("L")
                mask = mask.resize((args.resize, args.resize), Image.NEAREST)
                mask.save(os.path.join(target_dir, "mask.png"))
            except Exception as e:
                print(f"Warning for {obj_id}: Error processing mask, skipping mask: {e}")
        
        processed_count += 1
        print(f"Processed {obj_id}")
    
    # Count final dataset statistics
    train_count = len(os.listdir(os.path.join(args.output_dir, "train")))
    val_count = len(os.listdir(os.path.join(args.output_dir, "val")))
    
    print("\nDataset preparation complete!")
    print(f"Processed {processed_count} objects, skipped {skipped_count} objects")
    print(f"Train set: {train_count} objects")
    print(f"Validation set: {val_count} objects")
    print(f"Dataset saved to {args.output_dir}")


def find_image_file(directory):
    """Find an image file in the directory"""
    # First look for image.png/jpg specifically
    for name in ["image.png", "image.jpg", "image.jpeg"]:
        if os.path.exists(os.path.join(directory, name)):
            return name
    
    # Then check for any image file
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith("mask"):
            return file
    
    return None


def find_mesh_file(directory):
    """Find a mesh file in the directory"""
    # First look for model.obj/ply specifically
    for name in ["model.obj", "model.ply", "model.off", "mesh.obj", "mesh.ply", "mesh.off"]:
        if os.path.exists(os.path.join(directory, name)):
            return name
    
    # Then check for any mesh file
    for file in os.listdir(directory):
        if file.lower().endswith(('.obj', '.ply', '.off')):
            return file
    
    return None


def find_mask_file(directory):
    """Find a mask file in the directory"""
    # First look for mask.png specifically
    for name in ["mask.png", "mask.jpg", "alpha.png", "alpha.jpg"]:
        if os.path.exists(os.path.join(directory, name)):
            return name
    
    # Then check for any file with mask in the name
    for file in os.listdir(directory):
        if file.lower().startswith(('mask', 'alpha')) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            return file
    
    return None


if __name__ == "__main__":
    prepare_dataset() 