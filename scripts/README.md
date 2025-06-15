
# YOLOv12 Object Detection Training Guide

This guide provides instructions for training an object detection model using YOLOv12. The example below demonstrates how to fine-tune the YOLOv12n model. Pre-trained checkpoints are available for download from the Ultralytics Releases page. You can see more at this URL:
[Ultralytics Releases](https://github.com/ultralytics/assets/releases)

## Prerequisites

-   Ensure you have the Ultralytics YOLO package installed.
    
-   Download the desired YOLOv12 model checkpoint (e.g., yolo12n.pt) using the provided script.
    

## 1 Downloading Pre-trained Models

To download YOLOv12 model checkpoints, run the following command:

```bash
python scripts/download_yolo_model.py \
    --url <yolo_model_released_url> \
    --output-dir <saved_yolo_model_path>
```

This will save the pre-trained weights to the ./ckpts/raw/ directory.

## 2 Process Dataset
Here is the CLI command to download and process datasets.
```bash
python scripts/download_and_process_datasets.py \
    --output-dir <combined_dataset_path> \
    --dataset-base-dir <directory_containing_all_datasets> \
    --config <datasets_config_path> \
    --platforms <list_of_platforms_to_download_from> \  # e.g., ["kaggle", "roboflow", "huggingface"]
    --roboflow-api-key <roboflow_api_key>  # Optional: required if "roboflow" is included in --platforms
```
For example:
```bash
python scripts/download_and_process_datasets.py \
    --output-dir ./datasets/yolo_standard_dataset \
    --dataset-base-dir ./datasets/all_datasets \
    --config ./config/dataset_config.yaml \
    --roboflow-api-key YOUR_ROBOFLOW_APIKEY \
    --platforms "kaggle" "roboflow" "huggingface" # e.g., ["kaggle", "roboflow", "huggingface"]
```
For help:
```bash
python scripts/download_and_process_datasets.py -h
```
## 3 Fine-Tuning the Model
<!--
To fine-tune a YOLOv12 model for object detection, use the provided training script with customizable parameters. Run the following command and adjust the arguments based on your requirements:

```bash
python scripts/train_yolo.py \
    --epochs <number_of_epochs> \
    --batch <batch_size> \
    --device <cuda_device_id_or_list|cpu> \
    --project <path_to_save_results> \
    --name <project_name> \
    --resume  # Optional: resume training from the last checkpoint
```

### Example Configuration

For reference, the equivalent configuration using the yolo CLI command is shown below:

```bash
python scripts/train_yolo.py\
    --epochs 100 \
    --batch 32 \
    --device 0 \
    --project "./ckpts/finetune/runs" \
    --name "license_plate_detector"
```

### More Configurations
Run this CLI command to show `Help`.
```bash
python scripts/train_yolo.py -h
```
-->
To fine-tune a YOLOv12 model for object detection, use the provided training script with customizable parameters. Run the following command and adjust the arguments based on your requirements:

```bash
yolo detect train \
    model=<yolo_model_path or yolo_version_name> \
    data=<dataset_config_path> \
    epochs=<number_of_epochs> \
    batch=<batch_size> \
    patience=<early_stopping_patience> \
    imgsz=<image_size> \
    lr0=<initial_learning_rate> \
    lrf=<final_learning_rate> \
    device=<device_id or list_of_cuda or "cpu"> \
    project=<output_directory> \
    name=<experiment_name> \
    save=<save_checkpoint> \
    resume=<resume_training>
```

### Example Configuration

For reference, the equivalent configuration using the yolo CLI command is shown below:
```bash
yolo detect train \
    model="./ckpts/raw/yolo12n.pt" \
    data="./datasets/yolo_standard_dataset/data.yaml" \
    epochs=100 \
    batch=32 \
    patience=20 \
    imgsz=640 \
    lr0=0.01 \
    lrf=0.001 \
    device=0 \
    project="./ckpts/finetune/runs" \
    name="license_plate_detector" \
    save=true \
    resume=false
```
### More Configurations
Run this CLI command to show `Help`.
```bash
yolo --help
```

## Using PaddleOCR
