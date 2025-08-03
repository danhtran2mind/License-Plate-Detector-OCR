import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from inference.paddleocr_infer import process_ocr

# Append the current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def is_image_file(file_path):
    """Check if the file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return os.path.splitext(file_path)[1].lower() in image_extensions

def process_image(model, image_path, output_path=None):
    """Process a single image for license plate detection and OCR."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    try:
        results = model(image_path)
    except Exception as e:
        print(f"Error during image inference: {e}")
        return None, None
    
    plate_texts = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            
            # Crop the license plate region
            plate_region = image[y1:y2, x1:x2]
            # Run OCR on the cropped region
            ocr_results = process_ocr(plate_region)
            plate_text = ocr_results[0] if ocr_results else "No text detected"
            plate_texts.append(plate_text)
            
            # Draw bounding box and OCR text on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{plate_text} ({confidence:.2f})"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '_output.jpg'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to {output_path}")
    
    return image, plate_texts

def process_video(model, video_path, output_path=None):
    """Process a video for license plate detection and OCR, writing text on detected boxes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None, None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '_output.mp4'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = []
    all_plate_texts = []
    
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
        
        frame_plate_texts = []
        boxes_detected = False
        
        for result in results:
            for box in result.boxes:
                boxes_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                
                # Crop the license plate region
                plate_region = frame[y1:y2, x1:x2]
                
                # Run OCR on the cropped region
                ocr_results = process_ocr(plate_region)
                plate_text = ocr_results[0] if ocr_results else "No text detected"
                frame_plate_texts.append(plate_text)
                
                # Draw bounding box and OCR text on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{plate_text} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if boxes_detected:
            frames.append(frame)
            all_plate_texts.append(frame_plate_texts)
        else:
            # Append frame even if no boxes detected to maintain video continuity
            frames.append(frame)
            all_plate_texts.append([])
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Saved processed video to {output_path}")
    
    if not frames:
        print("No frames processed.")
        return None, None
    
    # Convert list of frames to 4D NumPy array
    video_array = np.stack(frames, axis=0)
    return video_array, all_plate_texts

def infer(input_path, output_path=None):
    """Main function to process either an image or video for license plate detection and OCR."""
    model_path = "ckpts/yolo/finetune/runs/license_plate_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return None, None
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    if is_image_file(input_path):
        result_array, plate_texts = process_image(model, input_path, output_path)
    else:
        result_array, plate_texts = process_video(model, video_path=input_path, output_path=output_path)
    
    return result_array, plate_texts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect and read license plates in an image or video.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or video file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output file (optional)")
    args = parser.parse_args()
    result_array, plate_texts = infer(args.input_path, args.output_path)