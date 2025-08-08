# Export ONNX Script Argument Documentation

This document describes the command-line arguments for the `export_onnx.py` script, which exports YOLOv8 models to ONNX format and verifies them.

## Usage
```bash
python export_onnx.py --model-paths <model_path> [options]
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-paths` | List of strings | Required | Paths to the YOLO model files (.pt). Accepts one or more model paths. |
| `--output-names` | List of strings | None | Output names for ONNX files. If not provided, defaults to the input model filename with `.onnx` extension. Must match the number of model paths if specified. |
| `--img-size` | Integer | 640 | Image size for model input (height and width, assuming square input). |
| `--dynamic` | Flag | False | Enable dynamic input shapes for the ONNX model. Use with `--dynamic` to activate. |
| `--simplify` | Flag | False | Simplify the ONNX model using the `onnx-simplifier` package. Use with `--simplify` to activate. |

## Examples

1. Export a single model with default settings:
```bash
python export_onnx.py --model-paths yolov8n.pt
```

2. Export multiple models with custom output names and dynamic shapes:
```bash
python export_onnx.py --model-paths yolov8n.pt yolov8s.pt --output-names model_n.onnx model_s.onnx --dynamic
```

3. Export a model with a specific image size and simplification:
```bash
python export_onnx.py --model-paths yolov8n.pt --img-size 416 --simplify
```

## Notes
- Ensure the number of `--output-names` matches the number of `--model-paths` if provided, or the script will exit with an error.
- The script verifies the exported ONNX model by running a dummy input through an ONNX inference session.
- The `--dynamic` flag enables dynamic input shapes, which can be useful for models that need to handle variable input sizes.
- The `--simplify` flag uses the `onnx-simplifier` package to optimize the ONNX model, potentially reducing its size and inference time.
