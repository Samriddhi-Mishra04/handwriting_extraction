# handwriting_extraction
# EasyOCR-Based Handwriting Text Extraction

## Overview
This script uses EasyOCR to extract handwritten or printed text from images while preserving original line breaks. Additionally, it provides a visual representation of detected text by drawing bounding boxes around recognized words.

## Features
- Uses EasyOCR for high-accuracy text extraction
- Preserves original line breaks in the extracted text
- Groups words into lines based on y-coordinate proximity
- Displays detected text with bounding boxes on the image

## Installation
Ensure that the necessary libraries are installed before running the script. If running in Google Colab, execute the following commands:

```bash
!pip install easyocr
!pip install opencv-python-headless
!pip install matplotlib
!pip install numpy
```

## Usage
1. Upload an image file containing handwritten or printed text.
2. The script extracts the text while preserving line breaks.
3. It then displays the image with bounding boxes around detected words.

## Code Explanation

### 1. Import Libraries
```python
import easyocr
import cv2
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np
from PIL import Image
```

### 2. Text Extraction Function
```python
def extract_text(image_path, languages=['en']):
    """
    Extracts text from an image using EasyOCR while preserving original line breaks.
    """
    reader = easyocr.Reader(languages)
    results = reader.readtext(image_path)
    results.sort(key=lambda x: x[0][0][1])  # Sort results by y-coordinate

    lines = []
    current_line = []
    threshold = 15  # Adjusting threshold for better grouping
    prev_y = None

    for bbox, text, prob in results:
        top_left_y = bbox[0][1]
        if prev_y is None or abs(top_left_y - prev_y) < threshold:
            current_line.append(text)
        else:
            lines.append(" ".join(current_line))
            current_line = [text]
        prev_y = top_left_y

    if current_line:
        lines.append(" ".join(current_line))
    
    return lines, results
```

### 3. Display Image with Bounding Boxes
```python
def display_image_with_boxes(image_path, results):
    """
    Displays the image with bounding boxes around detected text.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for bbox, text, prob in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
```

### 4. Upload and Process Image
```python
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
ocr_lines, ocr_results = extract_text(image_path, ['en'])

print("\nExtracted Text:")
for line in ocr_lines:
    print(line)

display_image_with_boxes(image_path, ocr_results)
```

## Output
- Extracted text is displayed in the console.
- The image with bounding boxes around detected text is shown.

## Requirements
- Python 3.6+
- EasyOCR
- OpenCV
- NumPy
- Matplotlib
- PIL

## Notes
- Adjust the `threshold` value in `extract_text()` to fine-tune line grouping.
- The script is designed for use in Google Colab but can be modified for local execution.

## License
This project is released under the MIT License.
