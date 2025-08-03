import gradio as gr
import os
from gradio_app.config import setup_logging, setup_sys_path
from gradio_app.processor import gradio_process, update_preview, update_visibility, clear_preview_data

# Initialize logging and sys.path
setup_logging()
setup_sys_path()

# Load custom CSS
custom_css = open(os.path.join(os.path.dirname(__file__), "gradio_app", "static", "styles.css"), "r").read()

# Define example files
examples = [
    {
        "input_file": os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "1", "lp_image.jpg"),
        "output_file": os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "1", "lp_image_output.jpg"),
        "input_type": "Image"
    },
    {
        "input_file": os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "2", "lp_video.mp4"),
        "output_file": os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "2", "lp_video_output.mp4"),
        "input_type": "Video"
    }
]

# Function to handle example selection
def load_example(evt: gr.SelectData):
    index = evt.index[0] if evt.index else 0
    example = examples[index]
    input_file = example["input_file"]
    output_file = example["output_file"]
    input_type = example["input_type"]
    
    # Update visibility based on input type
    input_preview_image, input_preview_video, output_image, output_video = update_visibility(input_type)
    
    # Update preview based on input file and type
    input_preview_image, input_preview_video = update_preview(input_file, input_type)
    
    return (
        input_file,
        input_type,
        input_preview_image,
        input_preview_video,
        output_file if input_type == "Image" else None,
        output_file if input_type == "Video" else None,
        "Example loaded - click Submit to process"
    )

# Gradio Interface
with gr.Blocks(css=custom_css) as iface:
    gr.Markdown(
        """
        # License Plate Detection and OCR
        Detect license plates from images or videos and read their text using 
        advanced computer vision and OCR for accurate identification.
        """,
        elem_classes="markdown-title"
    )
    gr.HTML("""
            You can explore the source code and contribute to the project on 
            <a href="https://github.com/danhtran2mind/License-Plate-Detector-OCR">danhtran2mind/License-Plate-Detector-OCR</a>.
            You can explore the HuggingFace Model Hub on 
            <a href="https://huggingface.co/danhtran2mind/license-plate-detector-ocr">danhtran2mind/license-plate-detector-ocr</a>.
    """)
        
    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(label="Upload Image or Video", elem_classes="custom-file-input")
            input_type = gr.Radio(choices=["Image", "Video"], label="Input Type", value="Image", elem_classes="custom-radio")
            with gr.Blocks():
                input_preview_image = gr.Image(label="Input Preview", visible=True, elem_classes="custom-image")
                input_preview_video = gr.Video(label="Input Preview", visible=False, elem_classes="custom-video")
            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary", elem_classes="custom-button secondary")
                submit_button = gr.Button("Submit", variant="primary", elem_classes="custom-button primary")
        with gr.Column(scale=1):
            with gr.Blocks():
                output_image = gr.Image(label="Processed Output (Image)", type="numpy", visible=True, elem_classes="custom-image")
                output_video = gr.Video(label="Processed Output (Video)", visible=False, elem_classes="custom-video")
            output_text = gr.Textbox(label="Detected License Plates", lines=10, elem_classes="custom-textbox")

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
    ).then(
        fn=clear_preview_data,
        inputs=None,
        outputs=None
    )

    # Examples table
    with gr.Row():
        gr.Markdown("### Examples")

    with gr.Row():
        example_table = gr.Dataframe(
            value=[[i, ex["input_type"], os.path.basename(ex["input_file"])] for i, ex in enumerate(examples)],
            headers=["Index", "Type", "File"],
            datatype=["number", "str", "str"],
            interactive=True,
            elem_classes="custom-table"
        )
    with gr.Row():
        gr.Markdown("""
                    This project utilizes:

                    - **Detection task**: YOLOv12 architecture model (YOLO12n) from [![GitHub Repo](https://img.shields.io/badge/GitHub-sunsmarterjie%2Fyolov12-blue?style=flat&logo=github)](https://github.com/sunsmarterjie/yolov12) and documentation at [![Ultralytics YOLO12](https://img.shields.io/badge/Ultralytics-YOLO12-purple?style=flat)](https://docs.ultralytics.com/models/yolo12/), powered by the Ultralytics platform: [![Ultralytics  Inc.](https://img.shields.io/badge/Ultralytics-Inc.-purple?style=flat)](https://docs.ultralytics.com).

                    - **OCR task**: PaddleOCR v2.9 from [![GitHub Repo](https://img.shields.io/badge/GitHub-PaddlePaddle%2FPaddleOCR%2Frelease%2F2.9-blue?style=flat&logo=github)](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.9), with the main repository at [![GitHub Repo](https://img.shields.io/badge/GitHub-PaddlePaddle%2FPaddleOCR-blue?style=flat&logo=github)](https://github.com/PaddlePaddle/PaddleOCR) for OCR inference. Explore more about PaddleOCR at [![PaddleOCR Website](https://img.shields.io/badge/PaddleOCR-Website-purple?style=flat)](https://www.paddleocr.ai/main/en/index.html).
                    """)
    # Example table click handler
    example_table.select(
        fn=load_example,
        inputs=None,
        outputs=[input_file, input_type, input_preview_image, input_preview_video, output_image, output_video, output_text]
    )

if __name__ == "__main__":
    iface.launch()