import pytesseract
from PIL import Image
import json

# Stage 1: Use Tesseract OCR to extract text from the image
def extract_text_with_ocr(image_path):
    """
    Extract text from image using Tesseract OCR
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Use Tesseract to extract text with Japanese support
        # --oem 3: Use both legacy and LSTM engines
        # --psm 6: Assume a single uniform block of text
        # -l jpn+eng: Use both Japanese and English language models
        extracted_text = pytesseract.image_to_string(image, config='-l jpn')
        
        return extracted_text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

# Stage 2: Optional - Use transformers model for structured extraction
def extract_structured_data(image_path):
    """
    Extract structured data using the DONUT model
    """
    try:
        from transformers import pipeline
        pipe = pipeline("image-text-to-text", model="mychen76/invoice-and-receipts_donut_v1")
        image = Image.open(image_path)
        result = pipe(image)
        return result
    except Exception as e:
        print(f"Error during structured extraction: {e}")
        return None

def main():
    image_path = "lesson4-receipt-reader/receipt.jpeg"
    
    print("=" * 50)
    print("STAGE 1: OCR TEXT EXTRACTION")
    print("=" * 50)
    
    # Extract raw text using Tesseract OCR
    ocr_text = extract_text_with_ocr(image_path)
    
    if ocr_text:
        print("Raw OCR Text:")
        print("-" * 30)
        print(ocr_text)
        print("-" * 30)
        print(f"Total characters extracted: {len(ocr_text)}")
    else:
        print("Failed to extract text using OCR")
        return
    
    print("\n" + "=" * 50)
    print("STAGE 2: STRUCTURED DATA EXTRACTION")
    print("=" * 50)
    
    # Extract structured data using transformers model
    structured_result = extract_structured_data(image_path)
    
    if structured_result:
        print("Structured extraction result:")
        print("-" * 30)
        if isinstance(structured_result, list) and len(structured_result) > 0:
            # Pretty print the result
            for item in structured_result:
                if isinstance(item, dict) and 'generated_text' in item:
                    print(item['generated_text'])
                else:
                    print(item)
        else:
            print(structured_result)
    else:
        print("Failed to extract structured data")

if __name__ == "__main__":
    main()