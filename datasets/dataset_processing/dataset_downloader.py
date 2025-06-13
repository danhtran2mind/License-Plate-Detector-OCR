"""Module for downloading license plate datasets using Roboflow API."""
from typing import List
import logging
from pathlib import Path
from roboflow import Roboflow
import os

def download_datasets(dataset_id_list: List[dict], api_key: str, output_dir: str) -> None:
    """Download datasets from Roboflow to the specified output directory.

    Args:
        dataset_id_list: List of dictionaries containing dataset details (e.g., {'workspace': str, 'project': str, 'version': int}).
        api_key: Roboflow API key for authentication.
        output_dir: Directory where datasets will be saved.
    """
    logging.info("Starting dataset download process.")
    
    # Initialize Roboflow client
    try:
        rf = Roboflow(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize Roboflow client: {e}")
        raise ValueError(f"Invalid Roboflow API key or connection error: {e}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download each dataset
    for idx, dataset_info in enumerate(dataset_id_list):
        try:
            workspace = dataset_info.get('workspace')
            project_name = dataset_info.get('project')
            version = dataset_info.get('version')

            if not all([workspace, project_name, version]):
                logging.warning(f"Skipping dataset {idx}: Missing required fields (workspace, project, version).")
                continue

            logging.info(f"Downloading dataset {idx}: {workspace}/{project_name}/{version}")
            project = rf.workspace(workspace).project(project_name)
            dataset = project.version(version).download(
                model_format="yolov8",  # Assuming YOLO-compatible format
                location=str(Path(output_dir) / f"dataset_{idx}"),
                overwrite=False
            )
            logging.info(f"Successfully downloaded dataset {idx} to {dataset.location}")

        except Exception as e:
            logging.error(f"Failed to download dataset {idx}: {e}")
            continue

    logging.info("Dataset download process completed.")