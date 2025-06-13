import subprocess
import uuid
from tqdm import tqdm
import json
import shutil
from pathlib import Path
import os
import yaml
import logging
import glob
import zipfile
import urllib.request
import glob
from roboflow import Roboflow
rf = Roboflow(api_key="7zE74v1Yo1YJxE9H5asu")

# Set up logging
logging.basicConfig(filename='dataset_conversion.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_and_remove_invalid_pairs(base_path, folder):
    label_dir = os.path.join(base_path, folder, 'labels')
    image_dir = os.path.join(base_path, folder, 'images')
    
    # Get all .txt files in the labels directory
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    for label_path in label_files:
        try:
            with open(label_path, 'r') as f:
                file_contents = f.read().strip()
                
                # Skip empty files
                if not file_contents:
                    print(f"Empty label file: {label_path}")
                    remove_pair(label_path, image_dir)
                    continue
                
                # Split the file into lines
                lines = file_contents.splitlines()
                
                # Check each line for exactly 5 elements
                for line in lines:
                    elements = line.strip().split()
                    if len(elements) != 5:
                        # print(f"Invalid label file (line with {len(elements)} elements): {label_path}")
                        remove_pair(label_path, image_dir)
                        break
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            remove_pair(label_path, image_dir)

def remove_pair(label_path, image_dir):
    # Get the base name (without extension) of the label file
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    
    # Construct the corresponding image path
    image_path = os.path.join(image_dir, f"{base_name}.jpg")
    
    # Remove the label file
    try:
        os.remove(label_path)
        # print(f"Removed label: {label_path}")
    except FileNotFoundError:
        print(f"Label file not found: {label_path}")
    
    # Remove the corresponding image file
    try:
        os.remove(image_path)
        # print(f"Removed image: {image_path}")
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")

def process_folders(base_path):
    for folder in ['train', 'valid', 'test']:
        print(f"\nProcessing {folder} folder...")
        check_and_remove_invalid_pairs(base_path, folder)

def convert_coco_huggingface_to_yolo(dataset_base_path, output_dir):
    """
    Convert COCO dataset to YOLOv11 format for train, val, and test datasets, adding UUID to filenames.

    Args:
        dataset_base_path (str): Base path to COCO dataset (train/val/test subdirs)
        output_dir (str): Output directory for YOLOv11 dataset
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

def create_yolo_structure(output_dir):
    """Create the YOLOv11 directory structure."""
    for d in ['train', 'valid', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, d, sub), exist_ok=True)
    logging.info(f"Created YOLOv11 directory structure at {output_dir}")

def copy_matched_files(src_image_dir, src_label_dir, dest_image_dir, dest_label_dir, split):
    """
    Copy matched image-label pairs to destination with UUID in filenames.

    Args:
        src_image_dir (str): Source directory for images
        src_label_dir (str): Source directory for labels
        dest_image_dir (str): Destination directory for images
        dest_label_dir (str): Destination directory for labels
        split (str): Dataset split (train, valid, test)

    Returns:
        tuple: (copied_files, img_mismatches, lbl_mismatches)
    """
    src_image_path = Path(src_image_dir)
    src_label_path = Path(src_label_dir)
    dest_image_path = Path(dest_image_dir)
    dest_label_path = Path(dest_label_dir)
    copied_files = set()

    # Get image and label files (case-insensitive)
    image_files = {}
    for ext in ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']:
        for f in src_image_path.glob(f'*.{ext}'):
            image_files[f.stem.lower()] = f
    label_files = {f.stem.lower(): f for f in src_label_path.glob('*.txt')}

    # Find matched pairs
    matched_stems = set(image_files.keys()) & set(label_files.keys())

    for stem in matched_stems:
        image_file = image_files[stem]
        label_file = label_files[stem]
        unique_id = str(uuid.uuid4())
        new_filename = f"{stem}_{split}_{unique_id}"

        # Copy files with UUID
        dest_image_file = dest_image_path / f"{new_filename}{image_file.suffix}"
        dest_label_file = dest_label_path / f"{new_filename}{label_file.suffix}"

        shutil.copy(image_file, dest_image_file)
        shutil.copy(label_file, dest_label_file)
        copied_files.add(dest_image_file.name)
        logging.info(f"Copied {image_file} to {dest_image_file}")
        logging.info(f"Copied {label_file} to {dest_label_file}")

    # Log mismatches
    images_without_labels = set(image_files.keys()) - matched_stems
    labels_without_images = set(label_files.keys()) - matched_stems
    for stem in images_without_labels:
        logging.warning(f"Image without label in {src_image_dir}: {image_files[stem]}")
    for stem in labels_without_images:
        logging.warning(f"Label without image in {src_label_dir}: {label_files[stem]}")

    return copied_files, len(images_without_labels), len(labels_without_images)

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

def coco_kaggle_to_yolo(output_dir="yolo_standard_dataset"):
    """
    Convert multiple Kaggle datasets to YOLOv11 format, adding UUID to filenames.

    Args:
        output_dir (str): Output directory for YOLOv11 dataset
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

    for dataset_name, splits in tqdm(datasets.items(), desc="Processing Kaggle Datasets"):
        for split in ['train', 'valid', 'test']:
            if split not in splits or not splits[split]:
                continue

            src_images = os.path.join(dataset_name, splits[split]['images'])
            dest_images = os.path.join(output_dir, split, 'images')
            src_labels = os.path.join(dataset_name, splits[split]['labels'])
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

# Main execution
dataset_id_list = [
    ("fareselmenshawii/large-license-plate-dataset", "kaggle"),
    ("duydieunguyen/licenseplates", "kaggle"),
    ("ronakgohil/license-plate-dataset", "kaggle"),
    ("bomaich/vnlicenseplate", "kaggle"),
    ("congtuu/vietnamese-license-plate-obb", "kaggle"),
    ("haitonthat/vietnam-license-plate-bounding-box", "kaggle"),
    ("university-of-southeastern-philippines-cnl9c/license-plate-detection-merged-projects", "roboflow", "yolov11", 3),
    ("ev-dshfb/license-plate-w8chc", "roboflow", "yolov11", 1),
    ("kanwal-masroor-gv4jr/yolov7-license-plate-detection", "roboflow", "yolov11", 3),
    ("keremberke/license-plate-object-detection", "huggingface")
]
os.makedirs("dataset", exist_ok=True)
os.chdir("/content/dataset")
combined_dataset_folder = "yolo_standard_dataset"
os.makedirs(combined_dataset_folder, exist_ok=True)

for data_no in tqdm(range(len(dataset_id_list)), desc="Download Datasets"):
    os.makedirs(f"dataset_{data_no}", exist_ok=True)
    os.chdir(f"dataset_{data_no}")

    if dataset_id_list[data_no][1] == "kaggle":
        dataset_id = dataset_id_list[data_no][0]
        dataset_name = dataset_id.split("/")[-1]
        urllib.request.urlretrieve(
            f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}",
            f"{dataset_name}.zip"
        )
        with zipfile.ZipFile(f"{dataset_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    elif dataset_id_list[data_no][1] == "roboflow":
        dataset_id = dataset_id_list[data_no][0]
        username = dataset_id.split("/")[0]
        dataset_name = dataset_id.split("/")[1]
        project = rf.workspace(username).project(dataset_name)
        version = project.version(dataset_id_list[data_no][-1])
        dataset = version.download(dataset_id_list[data_no][2])

        dataset_path = Path(dataset.location)
        if not dataset_path.exists():
            logging.error(f"Dataset path does not exist: {dataset_path}")

        current_dir = os.getcwd()
        logging.info(f"Current working directory: {current_dir}")
        logging.info(f"Dataset location: {dataset_path}")

        for folder in ['train', 'valid', 'test']:
            source_path = dataset_path / folder
            dest_path = Path("..") / combined_dataset_folder / folder

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
                        # unique_id = str(uuid.uuid4())
                        new_filename = f"{src_file.stem}_{folder}_{unique_id}{src_file.suffix}"
                        dest_file = dest_dir / new_filename

                        shutil.copy(src_file, dest_file)
                        logging.info(f"Copied {src_file} to {dest_file}")
                    except Exception as e:
                        logging.error(f"Failed to copy {src_file} to {dest_file}: {str(e)}")


    elif dataset_id_list[data_no][1] == "huggingface":
        dataset_id = dataset_id_list[data_no][0]
        subprocess.run(["git", "clone", f"https://huggingface.co/datasets/{dataset_id}"])
        os.chdir("license-plate-object-detection/data")
        for d in ["train", "valid", "test"]:
            os.makedirs(d, exist_ok=True)
        for z, d in zip(["train.zip", "test.zip", "valid.zip"], ["train", "valid", "test"]):
            if Path(z).exists():
                with zipfile.ZipFile(z, 'r') as zf:
                    zf.extractall(d)
        os.chdir("../..")

    os.chdir("..")

coco_kaggle_to_yolo(combined_dataset_folder)

convert_coco_huggingface_to_yolo(
    dataset_base_path=f"./dataset_{data_no}/license-plate-object-detection/data",
    output_dir=combined_dataset_folder)

process_folders("./yolo_standard_dataset")