import os
import cv2
import numpy as np
from uuid import uuid4
from inference.yolo_infer import yolo_infer
from inference.paddleocr_infer import process_ocr

def process_image(model_path, image_path, output_path=None):
    """Process a single image for license plate detection and OCR."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    try:
        results = yolo_infer(model_path, image_path)
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
    
    # Set default output path with UUID if not provided
    if output_path is None:
        output_dir = "apps/gradio_app/temp_data"
        output_path = os.path.join(output_dir, f"output_{uuid4()}.jpg")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to {output_path}")
    
    return image, plate_texts

def process_video(model_path, video_path, output_path=None):
    """Process a video for license plate detection and OCR."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return None, None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set default output path with UUID if not provided
    if output_path is None:
        output_dir = "apps/gradio_app/temp_data"
        output_path = os.path.join(output_dir, f"output_{uuid4()}.mp4")
    
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
            results = yolo_infer(model_path, frame)
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