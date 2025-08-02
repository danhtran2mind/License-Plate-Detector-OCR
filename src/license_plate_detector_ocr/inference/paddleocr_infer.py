# from paddleocr import PaddleOCR

# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     lang='en',
#     use_textline_orientation=False,  # This disables angle classification, so no need for use_angle_cls
#     text_recognition_model_name='en_PP-OCRv3_rec_slim',  # Fastest detection model
#     text_detection_model_name="en_PP-OCRv3_det_slim",
#     # rec_model_dir='ch_ppocr_mobile_v2.0_rec_infer',  # Fastest recognition model
# )

# result = ocr.ocr("plate-1.png", cls=True)

# plate_list = []
# for line in result:
#     current_line = []
#     for word_info in line:
#         current_line.append(word_info[-1][0])
#     plate_list.append(' '.join(current_line))
#####################3
from paddleocr import PaddleOCR
from typing import Union, List

# Initialize PaddleOCR once with optimized settings for English license plate recognition
OCR = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name='en_PP-OCRv3_det_slim',
    text_recognition_model_name='en_PP-OCRv3_rec_slim'
)

def process_ocr(image_input: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
    """
    Process OCR on a single image or a list of images.
    
    Args:
        image_input: A single image path (str) or a list of image paths (List[str])
        
    Returns:
        For a single image: List of extracted text strings
        For multiple images: List of lists, each containing extracted text strings for an image
    """
    # Convert single image path to a list for unified processing
    image_paths = [image_input] if isinstance(image_input, str) else image_input
    
    # Process each image and extract text
    results = []
    for path in image_paths:
        ocr_results = OCR.ocr(path, cls=False)  # cls=False since angle classification is disabled
        plate_list = [' '.join(word_info[-1][0] for word_info in line) for line in ocr_results if line]
        results.append(plate_list)
    
    # Return a single list for a single image, or list of lists for multiple images
    return results[0] if isinstance(image_input, str) else results

if __name__ == '__main__':
    # Example with a single image
    single_image = 'plate-1.png'
    single_result = process_ocr(single_image)
    print("Single image results:")
    print(single_result)
    for plate in single_result:
        print(plate)
    
    # Example with multiple images
    image_list = ['plate-1.png', 'plate-2.png', 'plate-3.jpg']
    multi_results = process_ocr(image_list)
    print("\nMultiple image results:")
    print(multi_results)
    for idx, plates in enumerate(multi_results):
        print(f"Image {idx + 1} ({image_list[idx]}):")
        for plate in plates:
            print(plate)