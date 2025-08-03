import logging
import os
import sys

def setup_logging():
    """Set up logging to a file for debugging."""
    logging.basicConfig(
        filename="apps/gradio_app/debug.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def setup_sys_path():
    """Adjust sys.path to include the src directory."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'license_plate_detector_ocr')))