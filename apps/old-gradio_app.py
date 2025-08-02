import gradio as gr
import os
import sys

# Adjust sys.path to include the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'license_plate_detector_ocr')))
from infer import infer, is_image_file

def gradio_process(input_file, input_type):
    """Process the input file (image or video) for license plate detection and OCR."""
    # Debugging: Print input file path
    print(f"Input file path: {input_file.name}")
    
    # Set default output path in apps/gradio_app/temp_data/
    output_dir = "apps/gradio_app/temp_data"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(input_file.name))[0] + ('_output.jpg' if is_image_file(input_file.name) else '_output.mp4')
    output_path = os.path.join(output_dir, output_filename)
    
    # Call the infer function from infer.py
    result_array, plate_texts = infer(input_file.name, output_path)
    
    if result_array is None:
        return None, f"Error: Processing failed for {input_file.name}"
    
    # Format plate texts for output
    if is_image_file(input_file.name):
        formatted_texts = "\n".join(plate_texts) if plate_texts else "No plates detected"
        return result_array, formatted_texts
    else:
        # For videos, plate_texts is a list of lists (per frame)
        formatted_texts = []
        for i, texts in enumerate(plate_texts):
            if texts:
                formatted_texts.append(f"Frame {i+1}: {', '.join(texts)}")
        formatted_texts = "\n".join(formatted_texts) if formatted_texts else "No plates detected"
        return output_path, formatted_texts

# Gradio Interface
iface = gr.Interface(
    fn=gradio_process,
    inputs=[
        gr.File(label="Upload Image or Video"),
        gr.Radio(choices=["Image", "Video"], label="Input Type", value="Image")
    ],
    outputs=[
        gr.Image(label="Processed Output", type="numpy"),
        gr.Textbox(label="Detected License Plates")
    ],
    title="License Plate Detection and OCR",
    description="Upload an image or video to detect and read license plates. Outputs are saved in apps/gradio_app/temp_data/."
)

if __name__ == "__main__":
    iface.launch(share=True)