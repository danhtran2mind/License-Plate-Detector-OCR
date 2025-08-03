import os
import sys
import shutil
import traceback
import logging
import gradio as gr
import uuid
import cv2
import time
from gradio_app.utils import convert_to_supported_format

# Adjust sys.path to include the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'license_plate_detector_ocr')))
from infer import infer, is_image_file

def gradio_process(input_file, input_type):
    """Process the input file (image or video) for license plate detection and OCR."""
    unique_id = str(uuid.uuid4())[:8]
    temp_input_dir = os.path.abspath(os.path.join("apps/gradio_app/temp_data", unique_id))
    preview_dir = os.path.abspath(os.path.join("apps/gradio_app/preview_data", unique_id))
    try:
        file_path = input_file.name if hasattr(input_file, 'name') else input_file
        logging.debug(f"Input file path: {file_path}")
        print(f"Input file path: {file_path}")
        
        # Verify source file exists and is readable
        if not os.path.exists(file_path):
            error_msg = f"Error: Source file {file_path} does not exist."
            logging.error(error_msg)
            return None, None, error_msg, None, None
        if not os.access(file_path, os.R_OK):
            error_msg = f"Error: Source file {file_path} is not readable."
            logging.error(error_msg)
            return None, None, error_msg, None, None
        
        # Create unique temp and preview directories
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)
        temp_input_path = os.path.join(temp_input_dir, os.path.basename(file_path))
        preview_input_path = os.path.join(preview_dir, os.path.basename(file_path))
        
        # Copy input file to temp and preview directories with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.copy2(file_path, temp_input_path)  # Copy to temp for processing
                shutil.copy2(file_path, preview_input_path)  # Copy to preview for display
                os.chmod(temp_input_path, 0o644)
                os.chmod(preview_input_path, 0o644)
                logging.debug(f"Copied input file to: {temp_input_path} and {preview_input_path}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    error_msg = f"Error copying file {file_path} to {temp_input_path} or {preview_input_path} after {max_retries} attempts: {str(e)}"
                    logging.error(error_msg)
                    return None, None, error_msg, None, None
                time.sleep(0.5)  # Brief delay before retry
        
        # Verify copied files
        for path in [temp_input_path, preview_input_path]:
            if not os.path.exists(path):
                error_msg = f"Error: Copied file {path} does not exist."
                logging.error(error_msg)
                return None, None, error_msg, None, None
            if not os.access(path, os.R_OK):
                error_msg = f"Error: Copied file {path} is not readable."
                logging.error(error_msg)
                return None, None, error_msg, None, None
            if os.path.getsize(path) == 0:
                error_msg = f"Error: Copied file {path} is empty."
                logging.error(error_msg)
                return None, None, error_msg, None, None
        
        # Validate image or video
        if is_image_file(temp_input_path):
            img = cv2.imread(temp_input_path)
            if img is None:
                error_msg = f"Error: Could not load image from {temp_input_path}."
                logging.error(error_msg)
                return None, None, error_msg, None, None
            # Check image properties
            height, width, channels = img.shape
            logging.debug(f"Image properties: {width}x{height}, {channels} channels")
            if channels not in (1, 3, 4):
                error_msg = f"Error: Unsupported number of channels ({channels}) in {temp_input_path}. Expected 1, 3, or 4."
                logging.error(error_msg)
                return None, None, error_msg, None, None
            if width == 0 or height == 0:
                error_msg = f"Error: Invalid image dimensions ({width}x{height}) in {temp_input_path}."
                logging.error(error_msg)
                return None, None, error_msg, None, None
        else:
            cap = cv2.VideoCapture(temp_input_path)
            if not cap.isOpened():
                error_msg = f"Error: Could not open video at {temp_input_path}."
                logging.error(error_msg)
                cap.release()
                return None, None, error_msg, None, None
            cap.release()
        
        # Set output path
        output_dir = os.path.abspath(os.path.join("apps/gradio_app/temp_data", str(uuid.uuid4())[:8]))
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{os.path.splitext(os.path.basename(temp_input_path))[0]}_{unique_id}_output{'_output.jpg' if is_image_file(temp_input_path) else '_output.mp4'}"
        output_path = os.path.join(output_dir, output_filename)
        logging.debug(f"Output path: {output_path}")
        
        # Call the infer function
        logging.debug(f"Calling infer with input: {temp_input_path}, output: {output_path}")
        result_array, plate_texts = infer(temp_input_path, output_path)
        
        if result_array is None and is_image_file(temp_input_path):
            error_msg = f"Error: Processing failed for {temp_input_path}. 'infer' returned None. Check infer.py logs for details."
            logging.error(error_msg)
            return None, None, error_msg, preview_input_path if is_image_file(temp_input_path) else None, preview_input_path if not is_image_file(temp_input_path) else None
        
        # Validate output file for videos
        if not is_image_file(temp_input_path):
            if not os.path.exists(output_path):
                error_msg = f"Error: Output video file {output_path} was not created."
                logging.error(error_msg)
                return None, None, error_msg, None, preview_input_path
            # Convert output video to supported format
            converted_output_path = os.path.join(output_dir, f"converted_{os.path.basename(output_path)}")
            converted_path = convert_to_supported_format(output_path, converted_output_path)
            if converted_path is None:
                error_msg = f"Error: Failed to convert output video {output_path} to supported format."
                logging.error(error_msg)
                return None, None, error_msg, None, preview_input_path
            output_path = converted_path
        
        # Format plate texts
        if is_image_file(temp_input_path):
            formatted_texts = "\n".join(plate_texts) if plate_texts else "No plates detected"
            logging.debug(f"Image processed successfully. Plate texts: {formatted_texts}")
            return result_array, None, formatted_texts, preview_input_path, None
        else:
            formatted_texts = []
            for i, texts in enumerate(plate_texts):
                if texts:
                    formatted_texts.append(f"Frame {i+1}: {', '.join(texts)}")
            formatted_texts = "\n".join(formatted_texts) if formatted_texts else "No plates detected"
            logging.debug(f"Video processed successfully. Plate texts: {formatted_texts}")
            return None, output_path, formatted_texts, None, preview_input_path
    except Exception as e:
        error_message = f"Error processing {file_path}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        print(error_message)
        return None, None, error_message, preview_input_path if is_image_file(file_path) else None, preview_input_path if not is_image_file(file_path) else None
    finally:
        # Clean up temp directory after processing, but keep preview directory
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            logging.debug(f"Cleaned up temporary directory: {temp_input_dir}")

def update_preview(file, input_type):
    """Return file path for the appropriate preview component based on input type."""
    if not file:
        logging.debug("No file provided for preview.")
        return None, None
    
    # Handle both file objects and string paths
    file_path = file.name if hasattr(file, 'name') else file
    logging.debug(f"Updating preview for {input_type}: {file_path}")
    
    # Verify file exists
    if not os.path.exists(file_path):
        logging.error(f"Input file {file_path} does not exist.")
        return None, None
    
    # Check if video format is supported
    if input_type == "Video" and not file_path.lower().endswith(('.mp4', '.webm')):
        logging.error(f"Unsupported video format for {file_path}. Use MP4 or WebM.")
        return None, None
    
    # Copy to preview directory for persistent display
    unique_id = str(uuid.uuid4())[:8]
    preview_dir = os.path.abspath(os.path.join("apps/gradio_app/preview_data", unique_id))
    os.makedirs(preview_dir, exist_ok=True)
    preview_input_path = os.path.join(preview_dir, os.path.basename(file_path))
    try:
        shutil.copy2(file_path, preview_input_path)
        os.chmod(preview_input_path, 0o644)
        logging.debug(f"Copied preview file to: {preview_input_path}")
    except Exception as e:
        logging.error(f"Error copying preview file to {preview_input_path}: {str(e)}")
        return None, None
    
    return preview_input_path if input_type == "Image" else None, preview_input_path if input_type == "Video" else None

def update_visibility(input_type):
    """Update visibility of input/output components based on input type."""
    logging.debug(f"Updating visibility for input type: {input_type}")
    is_image = input_type == "Image"
    is_video = input_type == "Video"
    return (
        gr.update(visible=is_image),
        gr.update(visible=is_video),
        gr.update(visible=is_image),
        gr.update(visible=is_video)
    )

def clear_preview_data():
    """Clear all files in the preview_data directory."""
    preview_data_dir = os.path.abspath("apps/gradio_app/preview_data")
    if os.path.exists(preview_data_dir):
        shutil.rmtree(preview_data_dir, ignore_errors=True)
        logging.debug(f"Cleared preview_data directory: {preview_data_dir}")
    os.makedirs(preview_data_dir, exist_ok=True)