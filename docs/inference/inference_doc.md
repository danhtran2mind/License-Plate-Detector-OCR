# License Plate Detection and OCR Inference Documentation

This document describes the inference process for the license plate detection and OCR system implemented in the provided Python scripts. The system uses a YOLO model for license plate detection and PaddleOCR for text recognition. Below are the details of the inference process and the arguments required to run the scripts.

## Overview

The system consists of two main scripts:
1. **`paddleocr_infer.py`**: Handles OCR processing for license plate text extraction.
2. **`infer.py`**: Manages the main inference pipeline, including license plate detection and OCR, for both images and videos.

The scripts process input images or videos, detect license plates using a YOLO model, crop the detected regions, and extract text using PaddleOCR. The output includes processed images or videos with bounding boxes and text annotations, along with the extracted license plate texts.

## Inference Process

### 1. `paddleocr_infer.py`

This script defines the OCR functionality using PaddleOCR, optimized for English license plate recognition.

#### Key Function: `process_ocr`
- **Purpose**: Extracts text from a single image, a list of images, or a NumPy array representing an image.
- **Input**:
  - `image_input`: Can be one of the following:
    - `str`: Path to a single image file.
    - `List[str]`: List of paths to multiple image files.
    - `np.ndarray`: A 3D NumPy array (height, width, channels) representing an image.
- **Output**:
  - For a single image or array: A list of extracted text strings (`List[str]`).
  - For multiple images: A list of lists, each containing extracted text strings for an image (`List[List[str]]`).
- **Behavior**:
  - Initializes PaddleOCR with English language settings and slim models for detection and recognition.
  - Processes input(s) and extracts text from detected regions.
  - Handles single or multiple inputs uniformly by converting single inputs to a list for processing.

#### Example Usage
```python
# Single image
result = process_ocr('<path_to_plate_image_1>')  # Returns List[str]

# Multiple images
results = process_ocr(['<path_to_plate_image_1>', '<path_to_plate_image_2>', '<path_to_plate_image_3>'])  # Returns List[List[str]]

# Image array
import cv2
image_array = cv2.imread('<path_to_plate_image>')
result = process_ocr(image_array)  # Returns List[str]
```

### 2. `infer.py`

This script integrates YOLO-based license plate detection with OCR to process images or videos.

#### Main Function: `infer`
- **Purpose**: Processes an input image or video to detect license plates and extract text.
- **Input Arguments**:
  - `input_path` (`str`, required): Path to the input image or video file.
  - `output_path` (`str`, optional): Path to save the processed output file. If not provided, defaults to the input path with `_output` appended (e.g., `input.jpg` → `input_output.jpg`).
- **Output**:
  - `result_array`: A NumPy array representing the processed image (3D for images, 4D for videos) or `None` if processing fails.
  - `plate_texts`: A list of extracted license plate texts (`List[str]` for images, `List[List[str]]` for videos) or `None` if processing fails.
- **Behavior**:
  - Loads a YOLO model from `ckpts/yolo/finetune/runs/license_plate_detector/weights/best.pt`.
  - Checks if the input is an image or video based on file extension.
  - Calls `process_image` for images or `process_video` for videos.
  - Saves the output with bounding boxes and text annotations.

#### Helper Functions
- **`is_image_file(file_path)`**:
  - Checks if a file has a valid image extension (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`).
  - Returns `True` for images, `False` otherwise.
- **`process_image(model, image_path, output_path)`**:
  - Processes a single image for license plate detection and OCR.
  - Draws bounding boxes and text with confidence scores on the image.
  - Saves the processed image to `output_path`.
  - Returns the processed image array and extracted texts.
- **`process_video(model, video_path, output_path)`**:
  - Processes a video frame by frame for license plate detection and OCR.
  - Draws bounding boxes and text with confidence scores on each frame.
  - Saves the processed video to `output_path`.
  - Returns a 4D NumPy array of frames and a list of per-frame extracted texts.

#### Command-Line Arguments
The script supports command-line execution with the following arguments:
## Command-Line Arguments
The script supports command-line execution with the following arguments:

| Argument        | Type | Required | Default | Description                                                                 | Example                                   |
|-----------------|------|----------|---------|-----------------------------------------------------------------------------|-------------------------------------------|
| `--model_path`  | `str` | None| `ckpts/yolo/finetune/runs/license_plate_detector/weights/best.pt`| Path to the model file.                                      | `--model_path <path_to_model_path>` or `--model_path best.pt` |
| `--input_path`  | `str` | Yes      | None    | Path to the input image or video file.                                      | `--input_path <path_to_plate_image_1>` or `--input_path video.mp4` |
| `--output_path` | `str` | No       | Input path with `_output` appended | Path to save the processed output file. | `--output_path output/plate_output.jpg` |

#### Example Command-Line Usage
```bash
# Process an image
python infer.py --input_path <path_to_plate_image_1> --output_path output/plate_output.jpg

# Process a video
python infer.py --input_path video.mp4 --output_path output/video_output.mp4
```

## Requirements
- **Python Libraries**:
  - `paddleocr`: For OCR processing.
  - `ultralytics`: For YOLO model inference.
  - `opencv-python` (`cv2`): For image and video processing.
  - `numpy`: For array operations.
- **Model File**:
  - YOLO model weights at `ckpts/yolo/finetune/runs/license_plate_detector/weights/best.pt`.
- **Input Files**:
  - Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`.
  - Videos: Any format supported by OpenCV (e.g., `.mp4`).

## Output Format
- **Images**:
  - Processed image saved with bounding boxes and text annotations.
  - Returns a 3D NumPy array (height, width, channels) and a list of extracted texts (`List[str]`).
- **Videos**:
  - Processed video saved with bounding boxes and text annotations on each frame.
  - Returns a 4D NumPy array (frames, height, width, channels) and a list of per-frame extracted texts (`List[List[str]]`).

## Error Handling
- Checks for the existence of the model file and input file.
- Validates image array dimensions (must be 3D).
- Handles failures in loading images/videos or during model inference, returning `None` for both outputs in case of errors.

## Notes
- The YOLO model and PaddleOCR are configured for English license plates. Modify `lang` or model names in `paddleocr_infer.py` for other languages.
- Ensure the model weights file exists at the specified path.
- Output directories are created automatically if they do not exist.
- For videos, frames without detected plates are included in the output to maintain continuity.
