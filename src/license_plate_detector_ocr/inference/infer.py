import os
import sys

# Append the current directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import paddleocr_infer

# Example with multiple images
image_list = ['plate-1.png', 'plate-2.png', 'plate-3.jpg']
multi_results = paddleocr_infer.process_ocr(image_list)
print("\nMultiple image results:")
print(multi_results)
for idx, plates in enumerate(multi_results):
    print(f"Image {idx + 1} ({image_list[idx]}):")
    for plate in plates:
        print(plate)