import gradio as gr
import os
from gradio_app.config import setup_logging, setup_sys_path
from gradio_app.processor import gradio_process, update_preview, update_visibility

# Initialize logging and sys.path
setup_logging()
setup_sys_path()

# Load custom CSS
custom_css = open(os.path.join(os.path.dirname(__file__), "gradio_app", "static", "styles.css"), "r").read()

# Define example files
example_image = os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "1", "lp_image.jpg")
example_image_output = os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "1", "lp_image_output.jpg")
example_video = os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "2", "lp_video.mp4")
example_video_output = os.path.join(os.path.dirname(__file__), "gradio_app", "assets", "examples", "license_plate_detector_ocr", "2", "lp_video_output.mp4")

# Format example files for Gradio File component
example_files = [
    {"path": example_image, "meta": {"_type": "gradio.FileData"}},
    {"path": example_video, "meta": {"_type": "gradio.FileData"}}
]

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
    
    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="Upload Image or Video",
                elem_classes="custom-file-input",
                file_types=["image", "video"],
                value=example_files
            )
            input_type = gr.Radio(
                choices=["Image", "Video"],
                label="Input Type",
                value="Image",
                elem_classes="custom-radio"
            )
            with gr.Blocks():
                input_preview_image = gr.Image(
                    label="Input Preview",
                    visible=True,
                    value=example_image,
                    elem_classes="custom-image"
                )
                input_preview_video = gr.Video(
                    label="Input Preview",
                    visible=False,
                    value=example_video,
                    elem_classes="custom-video"
                )
            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary", elem_classes="custom-button secondary")
                submit_button = gr.Button("Submit", variant="primary", elem_classes="custom-button primary")
        with gr.Column(scale=2):
            with gr.Blocks():
                output_image = gr.Image(
                    label="Processed Output (Image)",
                    type="numpy",
                    visible=True,
                    value=example_image_output,
                    elem_classes="custom-image"
                )
                output_video = gr.Video(
                    label="Processed Output (Video)",
                    visible=False,
                    value=example_video_output,
                    elem_classes="custom-video"
                )
            output_text = gr.Textbox(
                label="Detected License Plates",
                lines=10,
                elem_classes="custom-textbox"
            )

    # Update preview and output visibility when input type changes
    input_type.change(
        fn=update_visibility,
        inputs=input_type,
        outputs=[input_preview_image, input_preview_video, output_image, output_video]
    )

    # Update preview when file is uploaded
    input_file.change(
        fn=lambda file, input_type: (
            update_preview(file, input_type),
            "Image" if file and any(f["path"].lower().endswith(('.jpg', '.jpeg', '.png')) for f in file) else "Video"
        ),
        inputs=[input_file, input_type],
        outputs=[input_preview_image, input_preview_video, input_type]
    )
    
    # Bind the processing function
    submit_button.click(
        fn=gradio_process,
        inputs=[input_file, input_type],
        outputs=[output_image, output_video, output_text, input_preview_image, input_preview_video]
    )
    
    # Clear button functionality
    clear_button.click(
        fn=lambda: (
            None, None, None, "Image",
            example_image, example_video, example_image_output, example_video_output
        ),
        outputs=[
            input_file, output_image, output_video, input_type,
            input_preview_image, input_preview_video, output_image, output_video
        ]
    )

if __name__ == "__main__":
    iface.launch(share=True)