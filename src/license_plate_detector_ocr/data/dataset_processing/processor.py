import os
import glob
import logging

def check_and_remove_invalid_pairs(base_path, folder):
    label_dir = os.path.join(base_path, folder, 'labels')
    image_dir = os.path.join(base_path, folder, 'images')
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for label_path in label_files:
        try:
            with open(label_path, 'r') as f:
                file_contents = f.read().strip()
                if not file_contents:
                    remove_pair(label_path, image_dir)
                    continue
                lines = file_contents.splitlines()
                for line in lines:
                    elements = line.strip().split()
                    if len(elements) != 5:
                        remove_pair(label_path, image_dir)
                        break
        except Exception as e:
            logging.error(f"Error reading {label_path}: {e}")
            remove_pair(label_path, image_dir)

def remove_pair(label_path, image_dir):
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    image_path = os.path.join(image_dir, f"{base_name}.jpg")
    try:
        os.remove(label_path)
    except FileNotFoundError:
        logging.warning(f"Label file not found: {label_path}")
    try:
        os.remove(image_path)
    except FileNotFoundError:
        logging.warning(f"Image file not found: {image_path}")

def process_folders(base_path):
    for folder in ['train', 'valid', 'test']:
        logging.info(f"Processing {folder} folder...")
        check_and_remove_invalid_pairs(base_path, folder)
