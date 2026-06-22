#!/usr/bin/env python3
"""
Tesla Autopilot Clone - Preprocessing and KITTI to YOLO Converter Script
Organizes image folders and transforms rectangular coordinates [left, top, right, bottom]
into normalized center-relative ratios required by YOLOv8 networks.
"""

import os
import glob
import shutil
import random
import cv2
import yaml

# Seed random generators to establish deterministic train/val splits
random.seed(42)

# Source paths - configure these to match your local dataset folders
KITTI_IMAGES_DIR = "./original_kitti/images"
KITTI_LABELS_DIR = "./original_kitti/labels"
BASE_DIR = "./dataset"

# Taxon class mappings for output labels
CLASS_MAPPING = {
    'Car': 0, 'Van': 0,
    'Pedestrian': 1, 'Person_sitting': 1,
    'Cyclist': 2,
    'Truck': 3,
    'Tram': 4
}


def convert_kitti_to_yolo(bbox, img_width, img_height):
    """
    Transforms left, top, right, bottom pixel boundaries 
    into normalized YOLOv8 centering ratios (x_center, y_center, w, h)
    """
    left, top, right, bottom = bbox
    x_center = (left + right) / 2.0
    y_center = (top + bottom) / 2.0
    width = right - left
    height = bottom - top
    
    # Scale by pixel widths and heights
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # Clip absolute bounds between 0.0 and 1.0 to prevent floating exceptions
    return (
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(width, 0.0), 1.0),
        min(max(height, 0.0), 1.0)
    )


def setup_folders():
    """Generates standard nested YOLO folder layouts automatically"""
    dirs = [
        os.path.join(BASE_DIR, "images/train"),
        os.path.join(BASE_DIR, "images/val"),
        os.path.join(BASE_DIR, "labels/train"),
        os.path.join(BASE_DIR, "labels/val")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"[+] Directory active: {d}")


def main():
    print("=" * 60)
    print("Tesla Autopilot Clone Preprocessing Command Utility")
    print("=" * 60)
    
    # Verify source locations before proceeding
    if not os.path.exists(KITTI_IMAGES_DIR) or not os.path.exists(KITTI_LABELS_DIR):
        print(f"[-] Missing source directory: '{KITTI_IMAGES_DIR}' or '{KITTI_LABELS_DIR}' not detected.")
        print("    Please populate real annotations inside standard folders and retry.")
        return

    setup_folders()

    # Search for common image formats
    images = sorted(glob.glob(os.path.join(KITTI_IMAGES_DIR, "*.png"))) + \
             sorted(glob.glob(os.path.join(KITTI_IMAGES_DIR, "*.jpg")))

    if not images:
        print("[-] Error: Found 0 frame images in your source directory.")
        return

    print(f"[i] Located {len(images)} raw images. Splitting into 80/20 train/validation segments.")
    random.shuffle(images)
    split_idx = int(0.8 * len(images))
    
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    def process_split_set(files, split_name):
        copied_count = 0
        for img_path in files:
            img_name = os.path.basename(img_path)
            file_id, _ = os.path.splitext(img_name)
            label_path = os.path.join(KITTI_LABELS_DIR, f"{file_id}.txt")
            
            # Skip if label matching has failed
            if not os.path.exists(label_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            
            yolo_labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 15:
                        continue
                    
                    obj_type = parts[0]
                    if obj_type in CLASS_MAPPING:
                        class_id = CLASS_MAPPING[obj_type]
                        bbox = [float(x) for x in parts[4:8]]
                        yolo_coords = convert_kitti_to_yolo(bbox, w, h)
                        
                        coords_str = " ".join(f"{coord:.6f}" for coord in yolo_coords)
                        yolo_labels.append(f"{class_id} {coords_str}")
            
            # Write out translated assets
            dest_img = os.path.join(BASE_DIR, f"images/{split_name}/{img_name}")
            dest_txt = os.path.join(BASE_DIR, f"labels/{split_name}/{file_id}.txt")
            
            shutil.copy(img_path, dest_img)
            with open(dest_txt, 'w') as out_f:
                out_f.write("\n".join(yolo_labels))
            
            copied_count += 1
            
        print(f"[✓] Processed split [{split_name}]: Moved ({copied_count}/{len(files)}) elements")

    process_split_set(train_files, "train")
    process_split_set(val_files, "val")

    # Generate the dataset data.yaml file dynamically
    dataset_cfg = {
        'path': os.path.abspath(BASE_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'car',
            1: 'pedestrian',
            2: 'cyclist',
            3: 'truck',
            4: 'tram'
        }
    }
    
    yaml_dest = os.path.join(BASE_DIR, "data.yaml")
    with open(yaml_dest, 'w') as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)
        
    print(f"[+] Dataset data.yaml file written key paths: {yaml_dest}")
    print("=" * 60)
    print("Successful Precompilation. Ready to launch YOLOv8 training.")


if __name__ == "__main__":
    main()
