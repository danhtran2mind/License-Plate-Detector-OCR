# License-Plate-Detector-OCR🎟️ (License-Plate-Detector-OCR)



## Introduction

## Key Features

## Notebook



## Dataset

## Base Model

## Demonstration
Experience the magic of Ghibli-style video generation:  
- **HuggingFace Space**: [![HuggingFace Space Demo](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FText2Video--Ghibli--style-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/danhtran2mind/Text2Video-Ghibli-style)

- **Demo GUI**:  
  <img src="./assets/gradio_app_demo.jpg" alt="Gradio Demo" height="600">

To run the Gradio app locally (`localhost:7860`):  
```bash
python apps/gradio_app.py
```


## Usage Guide

### Setup Instructions

#### Step 1: Clone the Repository
Clone the project repository and navigate to the project directory:
```bash
git clone https://github.com/danhtran2mind/License-Plate-Detector-OCR.git
cd License-Plate-Detector-OCR
```

#### Step 2: Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements/requirements.txt
```

#### Step 3: Configure the Environment
Run the following scripts to set up the project:
- **Prepare Dataset (Optional, for Training)**  
  - Using `Kaggle`, `Roboflow`, `HuggingFace`
    ```bash
    python scripts/download_and_process_datasets.py --roboflow-api-key <your_roboflow_api_key>
    ```
  - Using `Kaggle`, `HuggingFace` only
    ```bash
    python scripts/download_and_process_datasets.py --platforms kaggle huggingface
    ```
  - Using `Roboflow` only
    ```bash
    python scripts/download_and_process_datasets.py --platforms roboflow --roboflow-api-key 
    <your_roboflow_api_key>
    ```
  
- **Download Model Checkpoints**  
  ```bash
  python scripts/download_ckpts.py
  ```
- Scripts Arguments
Refer to the [Scripts Documents](docs/scripts/scripts_doc.md) for detailed Scripts Arguments used in fine-tuning the model. ⚙️

### Training
- To train the model:
```bash
python src/license_plate_detector_ocr/training/train_yolo.py
```
- Training Arguments
Refer to the [Training Documents](docs/training/training_doc.md) for detailed Training Arguments used in fine-tuning the model. ⚙️

### Inference
To generate videos using the trained model:
```bash
python src/license_plate_detector_ocr/infer.py
```
- Inference Arguments
Refer to the [Inference Documents](docs/inference/inference_doc.md) for detailed Inference Arguments used in fine-tuning the model. ⚙️
## Inference Samples



## Environment
- **Python**: 3.10 or higher
- **Key Libraries**: See [requirements_compatible.txt](requirements/requirements_compatible.txt) for compatible versions
<!-- 
## Contact
For questions or issues, please use the [GitHub Issues tab](https://github.com/danhtran2mind/Text2Video-Ghibli-style/issues) or the [Hugging Face Community tab](https://huggingface.co/spaces/danhtran2mind/Text2Video-Ghibli-style/discussions). 📬 -->


<!-- 
## Project Description

This repository is trained from [![GitHub Repo](https://img.shields.io/badge/GitHub-danhtran2mind%2FMotionDirector-blue?style=flat&logo=github)](https://github.com/danhtran2mind/MotionDirector), a fork of [![GitHub Repo](https://img.shields.io/badge/GitHub-showlab%2FMotionDirector-blue?style=flat&logo=github)](https://github.com/showlab/MotionDirector), with numerous bug fixes and rewritten code for improved performance and stability. You can download the `zeroscope_v2_576w` model from the [![HuggingFace: cerspense/zeroscope_v2_576w](https://img.shields.io/badge/HuggingFace-cerspense%2Fzeroscope__v2__576w-yellow?logo=huggingface)](https://huggingface.co/cerspense/zeroscope_v2_576w). Explore more models on [![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-cerspense-yellow?style=flat&logo=huggingface)](https://huggingface.co/cerspense). -->