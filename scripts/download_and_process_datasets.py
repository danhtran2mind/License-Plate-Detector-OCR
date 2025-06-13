"""Entry point for downloading and processing license plate datasets."""
import argparse
import sys
from pathlib import Path

# Add datasets/ directory to sys.path to enable imports from dataset_processing
datasets_dir = str(Path(__file__).resolve().parents[1] / 'datasets')
if datasets_dir not in sys.path:
    sys.path.insert(0, datasets_dir)

from dataset_processing.utils import load_yaml_config, setup_logging
from dataset_processing.dataset_downloader import download_datasets
from dataset_processing.dataset_converter import convert_coco_to_yolo, convert_kaggle_to_yolo
from dataset_processing.validator import process_folders

def check_python_version():
    """Ensure Python version is 3.6 or higher."""
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)

def main():
    """Download and convert datasets to YOLOv11 format."""
    check_python_version()

    parser = argparse.ArgumentParser(description="License Plate Dataset Converter")
    parser.add_argument('--api-key', type=str, required=True, help="Roboflow API key")
    parser.add_argument('--config', type=str, default='config/datasets.yaml', help="Path to dataset config YAML")
    parser.add_argument('--output-dir', type=str, default='yolo_standard_dataset', help="Output directory for YOLO dataset")
    args = parser.parse_args()

    setup_logging()
    try:
        config = load_yaml_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key {e} in config file.")
        sys.exit(1)

    dataset_id_list = config.get('dataset_id_list')
    if not dataset_id_list:
        print("Error: 'dataset_id_list' is empty or not provided in config.")
        sys.exit(1)

    download_datasets(dataset_id_list, args.api_key, args.output_dir)
    convert_kaggle_to_yolo(args.output_dir)
    convert_coco_to_yolo(f"dataset_{len(dataset_id_list)-1}/license-plate-object-detection", args.output_dir)
    process_folders(args.output_dir)

if __name__ == "__main__":
    main()