import os
import urllib.request
import zipfile
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
from roboflow import Roboflow
import shutil
import uuid

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

def download_datasets(combined_dataset_folder, roboflow_api_key):
    """Download datasets from various sources and copy to combined folder."""
    rf = Roboflow(api_key=roboflow_api_key)
    
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
                os.chdir("..")
                continue

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