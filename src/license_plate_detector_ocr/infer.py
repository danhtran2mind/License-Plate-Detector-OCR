import os
import sys
import logging
import traceback
from inference.image_video_processor import process_image, process_video

# Append the current directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def is_image_file(file_path):
    """Check if the file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return os.path.splitext(file_path)[1].lower() in image_extensions

def infer(input_path, output_path=None):
    """Process an image or video for license plate detection and OCR."""
    model_path = "ckpts/yolo/finetune/runs/license_plate_detector/weights/best.pt"
    
    logging.debug(f"Starting inference for {input_path} with output {output_path}")
    
    if not os.path.exists(model_path):
        error_msg = f"Error: Model file not found at {model_path}"
        logging.error(error_msg)
        print(error_msg)
        return None, None
    
    if not os.path.exists(input_path):
        error_msg = f"Error: Input file not found at {input_path}"
        logging.error(error_msg)
        print(error_msg)
        return None, None
    
    try:
        if is_image_file(input_path):
            result_array, plate_texts = process_image(model_path, input_path, output_path)
        else:
            result_array, plate_texts = process_video(model_path, input_path, output_path)
        
        if result_array is None:
            error_msg = f"Error: Processing failed in {'process_image' if is_image_file(input_path) else 'process_video'} for {input_path}"
            logging.error(error_msg)
            print(error_msg)
            return None, None
        
        logging.debug(f"Inference successful: {len(plate_texts)} plates detected")
        return result_array, plate_texts
    except Exception as e:
        error_msg = f"Error during inference for {input_path}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        print(error_msg)
        return None, None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect and read license plates in an image or video.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or video file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output file (optional)")
    args = parser.parse_args()
    result_array, plate_texts = infer(args.input_path, args.output_path)