import urllib.request
import zipfile
import subprocess
from pathlib import Path
import logging
import os

def download_kaggle_dataset(dataset_id, output_dir):
    try:
        dataset_name = dataset_id.split("/")[-1]
        zip_path = output_dir / f"{dataset_name}.zip"
        output_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}",
            str(zip_path)
        )
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logging.info(f"Downloaded Kaggle dataset: {dataset_id}")
    except Exception as e:
        logging.error(f"Failed to download Kaggle dataset {dataset_id}: {str(e)}")

def download_roboflow_dataset(dataset_id, format_type, version, output_dir, api_key):
    try:
        from roboflow import Roboflow
        import shutil
        rf = Roboflow(api_key=api_key)
        username, dataset_name = dataset_id.split("/")
        project = rf.workspace(username).project(dataset_name)
        version_obj = project.version(version)
        dataset = version_obj.download(format_type)
        dataset_path = Path(dataset.location)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Move/copy all files and folders from dataset_path to output_dir
        for item in dataset_path.iterdir():
            dest = output_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        logging.info(f"Downloaded Roboflow dataset: {dataset_id} to {output_dir}")
        # Optionally, clean up the original download directory
        try:
            shutil.rmtree(dataset_path)
        except Exception as cleanup_e:
            logging.warning(f"Could not remove original Roboflow download dir {dataset_path}: {cleanup_e}")
    except Exception as e:
        logging.error(f"Failed to download Roboflow dataset {dataset_id}: {str(e)}")

def download_huggingface_dataset(dataset_id, output_dir):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", f"https://huggingface.co/datasets/{dataset_id}"], cwd=output_dir)
        data_dir = output_dir / dataset_id.split("/")[-1] / "data"
        for d in ["train", "valid", "test"]:
            (data_dir / d).mkdir(exist_ok=True)
        for z, d in zip(["train.zip", "test.zip", "valid.zip"], ["train", "valid", "test"]):
            zip_path = data_dir / z
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(data_dir / d)
        logging.info(f"Downloaded HuggingFace dataset: {dataset_id}")
    except Exception as e:
        logging.error(f"Failed to download HuggingFace dataset {dataset_id}: {str(e)}")
