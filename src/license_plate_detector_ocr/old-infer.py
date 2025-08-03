import os
import sys

# Append the current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from inference import paddleocr_infer

# Example with multiple images
image_list = ['plate-1.png', 'plate-2.png', 'plate-3.jpg']
multi_results = paddleocr_infer.process_ocr(image_list)
print("\nMultiple image results:")
print(multi_results)
for idx, plates in enumerate(multi_results):
    print(f"Image {idx + 1} ({image_list[idx]}):")
    for plate in plates:
        print(plate)


####yolo####
import cv2
from ultralytics import YOLO
import os
import argparse
import numpy as np

def is_image_file(file_path):
    """Check if the file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return os.path.splitext(file_path)[1].lower() in image_extensions

def process_image(model, image_path):
    """Process a single image for license plate detection and return the processed 3D array."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    try:
        results = model(image_path)
    except Exception as e:
        print(f"Error during image inference: {e}")
        return None
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"License Plate: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def process_video(model, video_path):
    """Process a video for license plate detection and return the processed 4D array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        try:
            results = model(frame)
        except Exception as e:
            print(f"Error during video inference: {e}")
            break
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"License Plate: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frames.append(frame)
    
    cap.release()
    if not frames:
        print("No frames processed.")
        return None
    
    # Convert list of frames to 4D NumPy array (num_frames, height, width, channels)
    video_array = np.stack(frames, axis=0)
    return video_array

def main(input_path):
    """Main function to process either an image or video for license plate detection."""
    model_path = "ckpts/yolo/finetune/runs/license_plate_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return None
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    if is_image_file(input_path):
        return process_image(model, input_path)
    else:
        return process_video(model, input_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect license plates in an image or video.")
    parser.add_argument("input_path", type=str, help="Path to the input image or video file")
    args = parser.parse_args()
    result = main(args.input_path)
    if result is not None:
        print(f"Processed array shape: {result.shape}")
    # _array = main("input_image.jpg")