import json
import os
import shutil
import uuid
import yaml
import logging
from pathlib import Path

def convert_coco_huggingface_to_yolo(dataset_base_path, output_dir):
    for dataset_type in ["train", "valid", "test"]:
        coco_path = Path(dataset_base_path) / dataset_type / "_annotations.coco.json"
        if not coco_path.exists():
            logging.info(f"Skipping {dataset_type}: {coco_path} not found")
            continue
        yolo_dir = Path(output_dir) / dataset_type
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        for dir_path in [yolo_dir, images_dir, labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        with open(coco_path) as f:
            coco_data = json.load(f)
        img_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        img_id_to_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            x_min, y_min, bbox_w, bbox_h = ann['bbox']
            width, height = img_id_to_dims[img_id]
            x_center = (x_min + bbox_w / 2) / width
            y_center = (y_min + bbox_h / 2) / height
            norm_w = bbox_w / width
            norm_h = bbox_h / height
            unique_id = str(uuid.uuid4())
            original_filename = Path(img_id_to_file[img_id]).stem
            new_filename = f"{original_filename}_{unique_id}"
            label_file = labels_dir / f"{new_filename}.txt"
            with open(label_file, 'a') as f:
                f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            src_img = Path(coco_path).parent / img_id_to_file[img_id]
            dst_img = images_dir / f"{new_filename}{Path(img_id_to_file[img_id]).suffix}"
            if src_img.exists() and not dst_img.exists():
                shutil.copy(src_img, dst_img)
                logging.info(f"Copied {src_img} to {dst_img}")
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

def create_yolo_structure(output_dir):
    for d in ['train', 'valid', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, d, sub), exist_ok=True)
    logging.info(f"Created YOLOv11 directory structure at {output_dir}")

def copy_matched_files(src_image_dir, src_label_dir, dest_image_dir, dest_label_dir, split):
    src_image_path = Path(src_image_dir)
    src_label_path = Path(src_label_dir)
    dest_image_path = Path(dest_image_dir)
    dest_label_path = Path(dest_label_dir)
    copied_files = set()
    image_files = {}
    for ext in ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']:
        for f in src_image_path.glob(f'*.{ext}'):
            image_files[f.stem.lower()] = f
    label_files = {f.stem.lower(): f for f in src_label_path.glob('*.txt')}
    matched_stems = set(image_files.keys()) & set(label_files.keys())
    for stem in matched_stems:
        image_file = image_files[stem]
        label_file = label_files[stem]
        unique_id = str(uuid.uuid4())
        new_filename = f"{stem}_{split}_{unique_id}"
        dest_image_file = dest_image_path / f"{new_filename}{image_file.suffix}"
        dest_label_file = dest_label_path / f"{new_filename}{label_file.suffix}"
        shutil.copy(image_file, dest_image_file)
        shutil.copy(label_file, dest_label_file)
        copied_files.add(dest_image_file.name)
        logging.info(f"Copied {image_file} to {dest_image_file}")
        logging.info(f"Copied {label_file} to {dest_label_file}")
    images_without_labels = set(image_files.keys()) - matched_stems
    labels_without_images = set(label_files.keys()) - matched_stems
    for stem in images_without_labels:
        logging.warning(f"Image without label in {src_image_dir}: {image_files[stem]}")
    for stem in labels_without_images:
        logging.warning(f"Label without image in {src_label_dir}: {label_files[stem]}")
    return copied_files, len(images_without_labels), len(labels_without_images)

def create_data_yaml(output_dir):
    data_yaml = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 1,
        'names': ['license_plate']
    }
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    logging.info("Created data.yaml")

def coco_kaggle_to_yolo(all_datasets_path, output_dir="yolo_standard_dataset"):
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
    from tqdm import tqdm
    for dataset_name, splits in tqdm(datasets.items(), desc="Processing Kaggle Datasets"):
        for split in ['train', 'valid', 'test']:
            if split not in splits or not splits[split]:
                continue
            src_images = os.path.join(all_datasets_path, dataset_name, splits[split]['images'])
            dest_images = os.path.join(output_dir, split, 'images')
            src_labels = os.path.join(all_datasets_path, dataset_name, splits[split]['labels'])
            dest_labels = os.path.join(output_dir, split, 'labels')
            copied_files, img_mismatches, lbl_mismatches = copy_matched_files(
                src_images, src_labels, dest_images, dest_labels, split
            )
            total_mismatches += img_mismatches + lbl_mismatches
            if img_mismatches > 0 or lbl_mismatches > 0:
                logging.warning(f"Mismatches in {dataset_name} {split} split: "
                               f"{img_mismatches} images without labels, {lbl_mismatches} labels without images")
    create_data_yaml(output_dir)
    logging.info(f"Dataset conversion completed. Total mismatches: {total_mismatches}")
    if total_mismatches > 0:
        logging.info("Check dataset_conversion.log for details on mismatched files")

def copy_dataset_to_combined_folder(dataset_path, combined_dataset_folder):
    # if not dataset_path.exists():
    #     logging.error(f"No child folders found in {dataset_path}. Please check the dataset structure.")
    #     return
    # child_folder_names = [p.name for p in dataset_path.iterdir() if p.is_dir()]
    for folder in ['train', 'valid', 'test']:
        # source_path = dataset_path / child_folder_names[0] / folder
        source_path = dataset_path / folder
        dest_path = Path(combined_dataset_folder) / folder
        if not source_path.exists():
            logging.warning(f"Source folder does not exist: {source_path}")
            continue
        unique_id = str(uuid.uuid4())
        for sub in ['images', 'labels']:
            src_dir = source_path / sub
            dest_dir = dest_path / sub
            if not src_dir.exists():
                logging.warning(f"Source directory does not exist: {src_dir}")
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_dir.glob('*'):
                try:
                    new_filename = f"{src_file.stem}_{folder}_{unique_id}{src_file.suffix}"
                    dest_file = dest_dir / new_filename
                    shutil.copy(src_file, dest_file)
                    logging.info(f"Copied {src_file} to {dest_file}")
                except Exception as e:
                    logging.error(f"Failed to copy {src_file} to {dest_file}: {str(e)}")
