
# YOLOv12 Object Detection Training Guide

This guide provides instructions for training an object detection model using YOLOv12. The example below demonstrates how to fine-tune the YOLOv12n model. Pre-trained checkpoints are available for download from the Ultralytics Releases page. You can see more at this URL:
[Ultralytics Releases](https://github.com/ultralytics/assets/releases)

## Prerequisites

-   Ensure you have the Ultralytics YOLO package installed.
    
-   Download the desired YOLOv12 model checkpoint (e.g., yolo12n.pt) using the provided script.
    

## Downloading Pre-trained Models

To download YOLOv12 model checkpoints, run the following command:

```bash
python scripts/download_yolo_models.py \
    --url <yolo_model_released_url> \
    --output-dir <saved_yolo_model_path>
```

This will save the pre-trained weights to the ./ckpts/raw/ directory.

## Fine-Tuning the Model

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
    --name "license_plate_detector" \
```
### More Configurations
```bash
python scripts/train_yolo.py -h
```
## Notes

-   Ensure the dataset specified in data.yaml is properly formatted and accessible.
    
-   Modify hyperparameters (e.g., epochs, batch, lr0) based on your dataset and computational resources.
    
-   Training logs and checkpoints will be saved in the ./ckpts/finetune/runs/license_plate_detector/ directory.
    

For additional details on YOLOv12 and advanced configurations, refer to the Ultralytics Documentation.