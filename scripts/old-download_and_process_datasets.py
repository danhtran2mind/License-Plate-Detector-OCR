import argparse
import logging
from pathlib import Path
import sys
import os

# Add parent directory to sys.path
# parent_dir = str(Path(__file__).resolve().parents[1])
# sys.path.insert(0, parent_dir)

# # Append datasets folder to sys.path
# datasets_dir = os.path.join(parent_dir, "datasets")
# sys.path.insert(0, datasets_dir)

# Append the current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                "src", "license_plate_detector_ocr", "data")))

from dataset_processing import config_loader, downloader, processor, converter

def main(args):
    logging.basicConfig(filename='dataset_conversion.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    config = config_loader.load_config(args.config)
    datasets = config['datasets']
    os.makedirs(args.dataset_base_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Download datasets
    for idx, ds in enumerate(datasets):
        if ds['type'] == 'kaggle' and 'kaggle' in args.platforms:
            downloader.download_kaggle_dataset(ds['id'], Path(args.dataset_base_dir) / f"dataset_{idx}")
        elif ds['type'] == 'roboflow' and 'roboflow' in args.platforms:
            downloader.download_roboflow_dataset(ds['id'], ds['format'], ds['version'], Path(args.dataset_base_dir) / f"dataset_{idx}", args.roboflow_api_key)
        elif ds['type'] == 'huggingface' and 'huggingface' in args.platforms:
            downloader.download_huggingface_dataset(ds['id'], Path(args.dataset_base_dir) / f"dataset_{idx}")

    # Convert and combine datasets
    converter.coco_kaggle_to_yolo(args.dataset_base_dir, args.output_dir)
    for idx, ds in enumerate(datasets):
        if ds['type'] == 'roboflow' and 'roboflow' in args.platforms:
            converter.copy_dataset_to_combined_folder(Path(args.dataset_base_dir) / f"dataset_{idx}", args.output_dir)
    for idx, ds in enumerate(datasets):
        if ds['type'] == 'huggingface' and 'huggingface' in args.platforms:
            converter.convert_coco_huggingface_to_yolo(
                dataset_base_path=Path(args.dataset_base_dir) / f"dataset_{idx}/license-plate-object-detection/data",
                output_dir=args.output_dir)
    processor.process_folders(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process license plate datasets.")
    parser.add_argument("--output-dir", default="./data/yolo_standard_dataset", help="Output directory for YOLOv11 dataset")
    parser.add_argument("--dataset-base-dir", default="./data/all_datasets", help="Base directory for downloaded datasets")
    parser.add_argument("--roboflow-api-key", required=True, help="Roboflow API key for downloading datasets")
    parser.add_argument("--config", default="./configs/datasets_config.yaml", help="Path to dataset config YAML")
    parser.add_argument("--platforms", nargs="*", default=["kaggle", "roboflow", "huggingface"], choices=["kaggle", "roboflow", "huggingface"], help="Platforms to download (default: all)")
    
    args = parser.parse_args()

    main(args)
