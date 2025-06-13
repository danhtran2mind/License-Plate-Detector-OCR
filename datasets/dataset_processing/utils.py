import os
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(filename='dataset_conversion.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_yolo_structure(output_dir):
    """Create the YOLOv11 directory structure."""
    for d in ['train', 'valid', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, d, sub), exist_ok=True)
    logging.info(f"Created YOLOv11 directory structure at {output_dir}")

def copy_matched_files(src_image_dir, src_label_dir, dest_image_dir, dest_label_dir, split, unique_id):
    """
    Copy matched image-label pairs to destination with UUID in filenames.
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