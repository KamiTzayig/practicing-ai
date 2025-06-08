import pytesseract
from PIL import Image
import cv2
import numpy as np

def preprocess_image_for_ocr(image_path):
    """
    Preprocess image to improve OCR accuracy
    """
    # Read image with OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(thresh)
    
    return processed_image

def extract_text_with_different_configs(image_path):
    """
    Try different Tesseract configurations for optimal text extraction
    """
    # Open original image
    original_image = Image.open(image_path)
    
    # Preprocess image
    processed_image = preprocess_image_for_ocr(image_path)
    
    # Different PSM (Page Segmentation Mode) configurations
    configs = {
        "Default (PSM 6)": '--oem 3 --psm 6',
        "Single text line (PSM 7)": '--oem 3 --psm 7',
        "Single word (PSM 8)": '--oem 3 --psm 8',
        "Single block (PSM 6) + digits": '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-¥$€£ ',
        "Auto page segmentation (PSM 3)": '--oem 3 --psm 3',
        "Raw line, no formatting (PSM 13)": '--oem 3 --psm 13'
    }
    
    results = {}
    
    print("TESTING ORIGINAL IMAGE:")
    print("=" * 50)
    
    for config_name, config in configs.items():
        try:
            text = pytesseract.image_to_string(original_image, config=config)
            results[f"Original - {config_name}"] = text.strip()
            print(f"\n{config_name}:")
            print("-" * 30)
            print(text.strip()[:200] + "..." if len(text.strip()) > 200 else text.strip())
        except Exception as e:
            print(f"Error with {config_name}: {e}")
    
    print("\n" + "=" * 50)
    print("TESTING PREPROCESSED IMAGE:")
    print("=" * 50)
    
    for config_name, config in configs.items():
        try:
            text = pytesseract.image_to_string(processed_image, config=config)
            results[f"Processed - {config_name}"] = text.strip()
            print(f"\n{config_name}:")
            print("-" * 30)
            print(text.strip()[:200] + "..." if len(text.strip()) > 200 else text.strip())
        except Exception as e:
            print(f"Error with {config_name}: {e}")
    
    return results

def extract_receipt_data(image_path):
    """
    Extract specific receipt data using optimized OCR
    """
    # Use the best configuration for receipt text
    image = Image.open(image_path)
    
    # Extract with configuration optimized for receipts
    config = '--oem 3 --psm 6'
    full_text = pytesseract.image_to_string(image, config=config)
    
    # Also get bounding box data for structured extraction
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    
    # Extract words with confidence > 60
    extracted_words = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        confidence = int(data['conf'][i])
        if confidence > 60:
            text = data['text'][i].strip()
            if text:
                extracted_words.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                })
    
    return full_text, extracted_words

def main():
    image_path = "receipt.jpeg"
    
    print("RECEIPT OCR ANALYSIS")
    print("=" * 60)
    
    # Test different configurations
    results = extract_text_with_different_configs(image_path)
    
    print("\n" + "=" * 60)
    print("STRUCTURED EXTRACTION WITH CONFIDENCE")
    print("=" * 60)
    
    # Get structured data
    full_text, words = extract_receipt_data(image_path)
    
    print(f"\nExtracted {len(words)} words with confidence > 60%")
    print("\nHigh-confidence words:")
    print("-" * 40)
    
    for word in sorted(words, key=lambda x: x['confidence'], reverse=True)[:20]:
        print(f"{word['text']:<15} (confidence: {word['confidence']}%)")
    
    print(f"\nFull extracted text:\n{'-'*30}")
    print(full_text)

if __name__ == "__main__":
    main() 