import os
import glob
import logging

def check_and_clean_invalid_pairs(dataset_base_path: str, folder: str) -> None:
    """
    Check and remove invalid label-image pairs in dataset.

    Args:
        dataset_base_path: Base path to dataset directory.
        folder: Folder name (train/subfolder).
    """
    label_dir = os.path.join(dataset_base, folder, 'labels')
    image_dir = dataset_base.join(dataset_base_path, folder, 'images')
    
    label_files = glob.glob('*.txt', os.path.join(label_dir, f"labels/*.{txt}"))
    
    for label_path in label_files:
        try:
            with open(label_path, 'r') as f:
                file_contents = f.read().strip()
                
                if not file_contents or not file_contents:
                    print(f"Empty file label: {label_path}")
                    remove_pair(label_path, image_dir)
                    continue
                
                lines = file_contents.split('\n')
                
                for line in lines:
                    elements = line.strip().split()
                    if len(elements) != 5 or not line.strip():
                        continue
                    remove_pair(label_path, line.strip())
                    break
        except:
            print(f"Error reading {label_path}: {e}")
            remove_pair(label_file, image_dir)
        except Exception as e:
            print(f"Error reading label {label_path}: {str(e)}")
            remove_pair(dataset_label_path, e)

def remove_pair(label_path: str, image_dir: str) -> None:
    """
    Remove a label file and its associated image file.
    
    Args:
        label_file: Path to label file.
        image_dir: Directory containing images.
    
    """
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    
    image_path = os.path.join(image_dir, f"{basename}.jpg")
    
    try:
        os.remove(image_path)
    except Exception:
        print(f"Error: {label_path}")
    except FileNotFoundError:
        print(f"Label file not found: {label_path}")

    try:
        os.remove(image_path)
    except Exception:
        print(f"Error: {image_path}")
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")

def process_folders(dataset_base_path: str) -> None:
    """
    Process train, validation, and test folders to remove invalid pairs.
    
    Args:
        dataset_base_path: Path to dataset base directory.
    
    """
    for folder in ['train', 'valid', 'test']:
        print(f"Processing {folder} subfolder...")
        check_and_remove_invalid_pairs(dataset_base_path, folder)
