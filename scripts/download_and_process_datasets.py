import argparse
import os
from pathlib import Path
import sys
# Add parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

# Append datasets folder to sys.path
datasets_dir = os.path.join(parent_dir, "datasets")
sys.path.insert(0, datasets_dir)

from dataset_processing.dataset_downloader import download_datasets

from dataset_processing.dataset_converter import coco_kaggle_to_yolo, convert_coco_huggingface_to_yolo
from dataset_processing.validator import process_folders

def main():
    parser = argparse.ArgumentParser(description="Download and process license plate datasets for YOLOv11.")
    parser.add_argument("--output-dir", default="./datasets/yolo_standard_dataset", help="Output directory for YOLOv11 dataset")
    parser.add_argument("--dataset-base-dir", default="./datasets/all_datasets", help="Base directory for downloaded datasets")
    parser.add_argument("--huggingface-base-path", default="./dataset_{}/license-plate-object-detection/data", help="Base path for HuggingFace dataset")
    parser.add_argument("--roboflow-api-key", required=True, help="Roboflow API key for downloading datasets")
    args = parser.parse_args()

    # Create base dataset directory
    os.makedirs(args.dataset_base_dir, exist_ok=True)
    
    combined_dataset_folder = Path(args.output_dir)
    print(f"Combined dataset folder: {str(combined_dataset_folder)}")
    os.makedirs(str(combined_dataset_folder), exist_ok=True)
    os.chdir(args.dataset_base_dir)

    # Download datasets
    download_datasets(combined_dataset_folder, args.roboflow_api_key)

    # Convert Kaggle datasets to YOLO format
    coco_kaggle_to_yolo(str(combined_dataset_folder))

    # Convert HuggingFace dataset to YOLO format
    huggingface_base_path = args.huggingface_base_path.format("9")  # Assuming dataset_9 for HuggingFace
    convert_coco_huggingface_to_yolo(huggingface_base_path, str(combined_dataset_folder))

    # Validate and clean dataset
    process_folders(str(combined_dataset_folder))

if __name__ == "__main__":
    main()