import os
import glob
import logging

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
                    logging.info(f"Empty label file: {label_path}")
                    remove_pair(label_path, image_dir)
                    continue
                
                # Split the file into lines
                lines = file_contents.splitlines()
                
                # Check each line for exactly 5 elements
                for line in lines:
                    elements = line.strip().split()
                    if len(elements) != 5:
                        logging.info(f"Invalid label file (line with {len(elements)} elements): {label_path}")
                        remove_pair(label_path, image_dir)
                        break
        except Exception as e:
            logging.error(f"Error reading {label_path}: {e}")
            remove_pair(label_path, image_dir)

def remove_pair(label_path, image_dir):
    # Get the base name (without extension) of the label file
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    
    # Construct the corresponding image path
    image_path = os.path.join(image_dir, f"{base_name}.jpg")
    
    # Remove the label file
    try:
        os.remove(label_path)
        logging.info(f"Removed label: {label_path}")
    except FileNotFoundError:
        logging.error(f"Label file not found: {label_path}")
    
    # Remove the corresponding image file
    try:
        os.remove(image_path)
        logging.info(f"Removed image: {image_path}")
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")

def process_folders(base_path):
    """Validate and clean YOLO dataset folders."""
    for folder in ['train', 'valid', 'test']:
        logging.info(f"Processing {folder} folder...")
        check_and_remove_invalid_pairs(base_path, folder)