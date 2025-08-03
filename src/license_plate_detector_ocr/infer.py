import os
import sys
from inference.image_video_processor import process_image, process_video

# Append the current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def is_image_file(file_path):
    """Check if the file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return os.path.splitext(file_path)[1].lower() in image_extensions

def infer(input_path, output_path=None):
    """Main function to process either an image or video for license plate detection and OCR."""
    model_path = "ckpts/yolo/finetune/runs/license_plate_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return None, None
    
    if is_image_file(input_path):
        result_array, plate_texts = process_image(model_path, input_path, output_path)
    else:
        result_array, plate_texts = process_video(model_path, video_path=input_path, output_path=output_path)
    
    return result_array, plate_texts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect and read license plates in an image or video.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or video file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output file (optional)")
    args = parser.parse_args()
    result_array, plate_texts = infer(args.input_path, args.output_path)