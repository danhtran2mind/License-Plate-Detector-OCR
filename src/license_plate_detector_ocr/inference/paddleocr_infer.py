from paddleocr import PaddleOCR
from typing import Union, List
import numpy as np

# Initialize PaddleOCR once with optimized settings for English license plate recognition
OCR = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name='en_PP-OCRv3_det_slim',
    text_recognition_model_name='en_PP-OCRv3_rec_slim'
)

def process_ocr(image_input: Union[str, List[str], np.ndarray]) -> Union[List[str], List[List[str]]]:
    """
    Process OCR on a single image path, a list of image paths, or a 3D image array.
    
    Args:
        image_input: A single image path (str), a list of image paths (List[str]), or a 3D NumPy array (np.ndarray)
        
    Returns:
        For a single image or array: List of extracted text strings
        For multiple images: List of lists, each containing extracted text strings for an image
    """
    # Convert single inputs to a list for unified processing
    if isinstance(image_input, str):
        image_inputs = [image_input]
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim != 3:
            raise ValueError("Image array must be 3-dimensional (height, width, channels)")
        image_inputs = [image_input]
    else:
        image_inputs = image_input
    
    # Process each image or array and extract text
    results = []
    for input_item in image_inputs:
        ocr_results = OCR.ocr(input_item, cls=False)  # cls=False since angle classification is disabled
        plate_list = [' '.join(word_info[-1][0] for word_info in line) for line in ocr_results if line]
        results.append(plate_list)
    
    # Return a single list for a single image/array, or list of lists for multiple images
    return results[0] if isinstance(image_input, (str, np.ndarray)) else results

if __name__ == '__main__':
    # Example with a single image path
    single_image = 'plate-1.png'
    single_result = process_ocr(single_image)
    print("Single image path results:")
    print(single_result)
    for plate in single_result:
        print(plate)
    
    # Example with multiple image paths
    image_list = ['plate-1.png', 'plate-2.png', 'plate-3.jpg']
    multi_results = process_ocr(image_list)
    print("\nMultiple image path results:")
    print(multi_results)
    for idx, plates in enumerate(multi_results):
        print(f"Image {idx + 1} ({image_list[idx]}):")
        for plate in plates:
            print(plate)
    
    # Example with a 3D image array (simulated)
    # Note: Replace this with actual image data in practice
    import cv2
    image_array = cv2.imread('lp_image.jpg')  # Load an image as a NumPy array
    if image_array is not None:
        array_result = process_ocr(image_array)
        print("\nSingle image array results:")
        print(array_result)
        for plate in array_result:
            print(plate)
    else:
        print("\nFailed to load image array for testing")