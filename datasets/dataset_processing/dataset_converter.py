import json
import shutil
import uuid
import yaml
import logging
from pathlib import Path

import sys

# Add parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

# Append datasets folder to sys.path
datasets_dir = os.path.join(parent_dir, "datasets")
sys.path.insert(0, datasets_dir)

from dataset_processing.utils import create_yolo_structure, copy_matched_files

def coco_kaggle_to_yolo(output_dir="yolo_standard_dataset"):
    """
    Convert multiple Kaggle datasets to YOLOv11 format, adding UUID to filenames.
    """
    datasets = {
        'dataset_0': {
            'train': {'images': 'images/train', 'labels': 'labels/train'},
            'valid': {'images': 'images/val', 'labels': 'labels/val'},
            'test': {'images': 'images/test', 'labels': 'labels/test'}
        },
        'dataset_1': {
            'train': {'images': 'images/train', 'labels': 'labels/train'},
            'valid': {'images': 'images/val', 'labels': 'labels/val'},
            'test': {}
        },
        'dataset_2': {
            'train': {'images': 'archive/images/train', 'labels': 'archive/labels/train'},
            'valid': {'images': 'archive/images/val', 'labels': 'archive/labels/val'},
            'test': {}
        },
        'dataset_3': {
            'train': {'images': 'train/images', 'labels': 'train/labels'},
            'valid': {'images': 'valid/images', 'labels': 'valid/labels'},
            'test': {'images': 'test/images', 'labels': 'test/labels'}
        },
        'dataset_4': {
            'train': {'images': 'train/images', 'labels': 'train/labels'},
            'valid': {'images': 'valid/images', 'labels': 'valid/labels'},
            'test': {}
        },
        'dataset_5': {
            'train': {'images': 'train/images', 'labels': 'train/labels'},
            'valid': {'images': 'valid/images', 'labels': 'valid/labels'},
            'test': {}
        }
    }

    create_yolo_structure(output_dir)
    total_mismatches = 0

    for dataset_name, splits in datasets.items():
        for split in ['train', 'valid', 'test']:
            if split not in splits or not splits[split]:
                continue

            src_images = os.path.join(dataset_name, splits[split]['images'])
            dest_images = os.path.join(output_dir, split, 'images')
            src_labels = os.path.join(dataset_name, splits[split]['labels'])
            dest_labels = os.path.join(output_dir, split, 'labels')
            unique_id = str(uuid.uuid4())

            copied_files, img_mismatches, lbl_mismatches = copy_matched_files(
                src_images, src_labels, dest_images, dest_labels, split, unique_id
            )

            total_mismatches += img_mismatches + lbl_mismatches
            if img_mismatches > 0 or lbl_mismatches > 0:
                logging.warning(f"Mismatches in {dataset_name} {split} split: "
                               f"{img_mismatches} images without labels, {lbl_mismatches} labels without images")

    create_data_yaml(output_dir)
    logging.info(f"Dataset conversion completed. Total mismatches: {total_mismatches}")
    if total_mismatches > 0:
        logging.info("Check dataset_conversion.log for details on mismatched files")

def convert_coco_huggingface_to_yolo(dataset_base_path, output_dir):
    """
    Convert COCO dataset to YOLOv11 format for train, val, and test datasets, adding UUID to filenames.
    """
    for dataset_type in ["train", "valid", "test"]:
        coco_path = Path(dataset_base_path) / dataset_type / "_annotations.coco.json"
        if not coco_path.exists():
            logging.info(f"Skipping {dataset_type}: {coco_path} not found")
            continue

        # Create output directories
        yolo_dir = Path(output_dir) / dataset_type
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        for dir_path in [yolo_dir, images_dir, labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load COCO annotations
        with open(coco_path) as f:
            coco_data = json.load(f)

        # Image mappings
        img_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        img_id_to_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

        # Process annotations
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            x_min, y_min, bbox_w, bbox_h = ann['bbox']
            width, height = img_id_to_dims[img_id]

            # Convert to YOLO format
            x_center = (x_min + bbox_w / 2) / width
            y_center = (y_min + bbox_h / 2) / height
            norm_w = bbox_w / width
            norm_h = bbox_h / height

            # Generate UUID for unique filename
            unique_id = str(uuid.uuid4())
            original_filename = Path(img_id_to_file[img_id]).stem
            new_filename = f"{original_filename}_{unique_id}"

            # Write label
            label_file = labels_dir / f"{new_filename}.txt"
            with open(label_file, 'a') as f:
                f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            # Copy image with UUID
            src_img = Path(coco_path).parent / img_id_to_file[img_id]
            dst_img = images_dir / f"{new_filename}{Path(img_id_to_file[img_id]).suffix}"
            if src_img.exists() and not dst_img.exists():
                shutil.copy(src_img, dst_img)
                logging.info(f"Copied {src_img} to {dst_img}")

    # Create data.yaml
    yaml_content = {
        "path": str(Path(output_dir).absolute()),
        "train": "train/images",
        "valid": "valid/images",
        "test": "test/images",
        "names": {0: "license_plate"}
    }
    yaml_path = Path(output_dir) / "data.yaml"
    if not yaml_path.exists():
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        logging.info(f"Created {yaml_path}")

def create_data_yaml(output_dir):
    """Create the data.yaml file for YOLOv11."""
    data_yaml = {
        'path': './dataset',
        'train': './dataset/train/images',
        'val': './dataset/valid/images',
        'test': './dataset/test/images',
        'nc': 1,
        'names': ['license_plate']
    }
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    logging.info("Created data.yaml")