import gradio as gr
import os
import sys
import traceback
import logging
import shutil
import ffmpeg

# Set up logging to a file for debugging
logging.basicConfig(
    filename="apps/gradio_app/debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Adjust sys.path to include the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'license_plate_detector_ocr')))
from infer import infer, is_image_file

def convert_to_supported_format(input_path, output_path):
    """Convert video to a browser-compatible format (MP4 with H.264 codec)."""
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec='h264', acodec='aac', format='mp4', loglevel='quiet')
        ffmpeg.run(stream)
        logging.debug(f"Converted video to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error converting video {input_path}: {str(e)}")
        return None

def gradio_process(input_file, input_type):
    """Process the input file (image or video) for license plate detection and OCR."""
    try:
        logging.debug(f"Input file path: {input_file.name}")
        print(f"Input file path: {input_file.name}")
        
        # Copy input file to temp_data directory to ensure stability
        temp_input_dir = "apps/gradio_app/temp_data"
        os.makedirs(temp_input_dir, exist_ok=True)
        temp_input_path = os.path.join(temp_input_dir, os.path.basename(input_file.name))
        shutil.copy(input_file.name, temp_input_path)
        logging.debug(f"Copied input file to: {temp_input_path}")
        
        # Verify input file exists
        if not os.path.exists(temp_input_path):
            error_msg = f"Error: Input file {temp_input_path} does not exist."
            logging.error(error_msg)
            return None, None, error_msg, None, None
        
        # Set output path
        output_dir = "apps/gradio_app/temp_data"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.splitext(os.path.basename(temp_input_path))[0] + ('_output.jpg' if is_image_file(temp_input_path) else '_output.mp4')
        output_path = os.path.join(output_dir, output_filename)
        logging.debug(f"Output path: {output_path}")
        
        # Call the infer function
        result_array, plate_texts = infer(temp_input_path, output_path)
        
        if result_array is None and is_image_file(temp_input_path):
            error_msg = f"Error: Processing failed for {temp_input_path}. 'infer' returned None."
            logging.error(error_msg)
            return None, None, error_msg, None, None
        
        # Validate output file for videos
        if not is_image_file(temp_input_path):
            if not os.path.exists(output_path):
                error_msg = f"Error: Output video file {output_path} was not created."
                logging.error(error_msg)
                return None, None, error_msg, None, None
            # Convert output video to supported format
            converted_output_path = os.path.join(output_dir, f"converted_{os.path.basename(output_path)}")
            converted_path = convert_to_supported_format(output_path, converted_output_path)
            if converted_path is None:
                error_msg = f"Error: Failed to convert output video {output_path} to supported format."
                logging.error(error_msg)
                return None, None, error_msg, None, None
            output_path = converted_path
        
        # Format plate texts
        if is_image_file(temp_input_path):
            formatted_texts = "\n".join(plate_texts) if plate_texts else "No plates detected"
            logging.debug(f"Image processed successfully. Plate texts: {formatted_texts}")
            return result_array, None, formatted_texts, temp_input_path, None
        else:
            formatted_texts = []
            for i, texts in enumerate(plate_texts):
                if texts:
                    formatted_texts.append(f"Frame {i+1}: {', '.join(texts)}")
            formatted_texts = "\n".join(formatted_texts) if formatted_texts else "No plates detected"
            logging.debug(f"Video processed successfully. Plate texts: {formatted_texts}")
            return None, output_path, formatted_texts, None, temp_input_path
    except Exception as e:
        error_message = f"Error processing {input_file.name}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        print(error_message)
        return None, None, error_message, None, None

def update_preview(file, input_type):
    """Return file path for the appropriate preview component based on input type."""
    if not file:
        logging.debug("No file provided for preview.")
        return None, None
    logging.debug(f"Updating preview for {input_type}: {file.name}")
    # Verify file exists
    if not os.path.exists(file.name):
        logging.error(f"Input file {file.name} does not exist.")
        return None, None
    # Check if video format is supported
    if input_type == "Video" and not file.name.lower().endswith(('.mp4', '.webm')):
        logging.error(f"Unsupported video format for {file.name}. Use MP4 or WebM.")
        return None, None
    return file.name if input_type == "Image" else None, file.name if input_type == "Video" else None

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

# Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # License Plate Detection and OCR
        Upload an image or video to detect and read license plates. Outputs are saved in `apps/gradio_app/temp_data/`.
        Debug logs are saved in `apps/gradio_app/debug.log`.
        """,
        elem_classes="markdown-title"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(label="Upload Image or Video")
            input_type = gr.Radio(choices=["Image", "Video"], label="Input Type", value="Image")
            with gr.Blocks():
                input_preview_image = gr.Image(label="Input Preview", visible=True)
                input_preview_video = gr.Video(label="Input Preview", visible=False)
            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")
                submit_button = gr.Button("Submit", variant="primary")
        with gr.Column(scale=2):
            with gr.Blocks():
                output_image = gr.Image(label="Processed Output (Image)", type="numpy", visible=True)
                output_video = gr.Video(label="Processed Output (Video)", visible=False)
            output_text = gr.Textbox(label="Detected License Plates", lines=10)

    # Update preview and output visibility when input type changes
    input_type.change(
        fn=update_visibility,
        inputs=input_type,
        outputs=[input_preview_image, input_preview_video, output_image, output_video]
    )

    # Update preview when file is uploaded
    input_file.change(
        fn=update_preview,
        inputs=[input_file, input_type],
        outputs=[input_preview_image, input_preview_video]
    )
    
    # Bind the processing function
    submit_button.click(
        fn=gradio_process,
        inputs=[input_file, input_type],
        outputs=[output_image, output_video, output_text, input_preview_image, input_preview_video]
    )
    
    # Clear button functionality
    clear_button.click(
        fn=lambda: (None, None, None, "Image", None, None, None, None),
        outputs=[input_file, output_image, output_video, input_type, input_preview_image, input_preview_video, output_image, output_video]
    )

if __name__ == "__main__":
    iface.launch(share=True)