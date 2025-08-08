%%writefile export/export_onnx.py
import argparse
import os
import argparse
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np

# Function to export a model to ONNX
def export_to_onnx(model_path, output_name, img_size, dynamic, simplify):
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found.")
            return False

        # Load the YOLOv8 model
        model = YOLO(model_path)

        # Export to ONNX
        model.export(format="onnx", imgsz=img_size, dynamic=dynamic, simplify=simplify)
        print(f"Exported {model_path} to {output_name}")

        # Verify ONNX file was created
        onnx_path = os.path.join(os.path.dirname(model_path), output_name)
        if os.path.exists(onnx_path):
            print(f"ONNX file created successfully at {onnx_path}")
            return onnx_path
        else:
            print(f"Error: ONNX file {onnx_path} not created.")
            return False
    except Exception as e:
        print(f"Error during export of {model_path}: {str(e)}")
        return False

# Function to verify ONNX model
def verify_onnx_model(onnx_path, img_size):
    try:
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # Create dummy input (batch=1, channels=3, height=img_size, width=img_size)
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        print(f"ONNX model {onnx_path} verified successfully!")
        return True
    except Exception as e:
        print(f"Error verifying {onnx_path}: {str(e)}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Export YOLOv8 models to ONNX and verify them.")
    parser.add_argument('--model-paths', nargs='+', required=True, 
                        help="Paths to the YOLO model files (.pt)")
    parser.add_argument('--output-names', nargs='+', default=None,
                        help="Output names for ONNX files (default: same as input with .onnx extension)")
    parser.add_argument('--img-size', type=int, default=640,
                        help="Image size for model input (default: 640)")
    parser.add_argument('--dynamic', action='store_true',
                        help="Enable dynamic input shapes for ONNX model")
    parser.add_argument('--simplify', action='store_true',
                        help="Simplify the ONNX model using onnx-simplifier")

    # Parse arguments
    args = parser.parse_args()

    # Ensure output names match the number of model paths
    if args.output_names and len(args.output_names) != len(args.model_paths):
        print("Error: Number of output names must match number of model paths.")
        return

    # Set default output names if not provided
    output_names = args.output_names if args.output_names else [
        os.path.basename(path).replace(".pt", ".onnx") for path in args.model_paths
    ]

    # Process each model
    for model_path, output_name in zip(args.model_paths, output_names):
        print(f"\nExporting {model_path} to ONNX...")
        onnx_path = export_to_onnx(
            model_path, output_name, args.img_size, args.dynamic, args.simplify
        )

        if onnx_path:
            print(f"\nVerifying {output_name}...")
            verify_onnx_model(onnx_path, args.img_size)

if __name__ == "__main__":
    main()
