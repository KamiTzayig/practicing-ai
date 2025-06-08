import pytesseract
from PIL import Image
import re

def extract_receipt_text(image_path, lang='jpn+eng'):
    """
    Extract text from receipt image using optimized Tesseract OCR
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Use Tesseract with optimal configuration for receipt text
        # --oem 3: Use both legacy and LSTM engines
        # --psm 6: Assume a single uniform block of text
        # -l jpn+eng: Use both Japanese and English language models
        config = f'--oem 3 --psm 6 -l {lang}'
        
        # Extract text
        extracted_text = pytesseract.image_to_string(image, config=config)
        
        return extracted_text.strip()
    
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return None

def extract_receipt_text_multiple_langs(image_path):
    """
    Try different language combinations for optimal results
    """
    # Different language combinations to try
    language_configs = [
        ('jpn+eng', 'Japanese + English (Recommended for Japanese receipts)'),
        ('jpn', 'Japanese only'),
        ('eng', 'English only'),
        ('osd', 'Auto-detect orientation and script')
    ]
    
    results = {}
    
    for lang_code, description in language_configs:
        try:
            if lang_code == 'osd':
                # Special handling for orientation detection
                image = Image.open(image_path)
                config = '--psm 0'
                text = pytesseract.image_to_string(image, config=config)
            else:
                text = extract_receipt_text(image_path, lang_code)
            
            results[lang_code] = {
                'text': text,
                'description': description,
                'length': len(text) if text else 0
            }
            
        except Exception as e:
            results[lang_code] = {
                'text': f"Error: {e}",
                'description': description,
                'length': 0
            }
    
    return results

def extract_key_information(text):
    """
    Extract key receipt information from raw OCR text
    """
    if not text:
        return {}
    
    info = {}
    
    # Extract phone number (Japanese format)
    phone_pattern = r'TEL:\s*(\d{2,3}-\d{4}-\d{4})'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        info['phone'] = phone_match.group(1)
    
    # Extract date and time
    date_time_pattern = r'(\d{4})\s*(\d{1,2}[A-Z])\s*(\d{1,2}[A-Z])\s*.*?(\d{2}:\d{2})'
    date_time_match = re.search(date_time_pattern, text)
    if date_time_match:
        info['year'] = date_time_match.group(1)
        info['time'] = date_time_match.group(4)
    
    # Extract receipt/transaction numbers
    receipt_numbers = re.findall(r'#(\d{6})', text)
    if receipt_numbers:
        info['receipt_numbers'] = receipt_numbers
    
    # Extract prices (¥ symbol followed by numbers)
    prices = re.findall(r'¥(\d+(?:,\d{3})*)', text)
    if prices:
        # Convert to integers, removing commas
        info['prices'] = [int(price.replace(',', '')) for price in prices]
        info['total_found'] = max(info['prices']) if info['prices'] else 0
    
    # Extract percentages (for tax, discounts)
    percentages = re.findall(r'(\d+)%', text)
    if percentages:
        info['percentages'] = [int(p) for p in percentages]
    
    return info

def main():
    image_path = "receipt.jpeg"
    
    print("�� RECEIPT OCR READER (JAPANESE SUPPORT)")
    print("=" * 60)
    
    # First, try different language configurations
    print("🌐 TESTING DIFFERENT LANGUAGE CONFIGURATIONS:")
    print("=" * 60)
    
    lang_results = extract_receipt_text_multiple_langs(image_path)
    
    best_result = None
    best_length = 0
    
    for lang_code, result in lang_results.items():
        print(f"\n📝 {result['description']}:")
        print(f"   Language code: {lang_code}")
        print(f"   Characters extracted: {result['length']}")
        if result['length'] > 0 and lang_code != 'osd':
            print(f"   Sample: {result['text'][:100]}...")
            
            # Keep track of the best result
            if result['length'] > best_length:
                best_result = result['text']
                best_length = result['length']
    
    # Use the best result for detailed extraction
    if best_result:
        print(f"\n🏆 USING BEST RESULT ({best_length} characters):")
        print("=" * 60)
        
        print("✅ Raw OCR Text Extracted:")
        print("-" * 30)
        print(best_result)
        
        print("\n" + "=" * 60)
        print("🔍 KEY INFORMATION EXTRACTED:")
        print("=" * 60)
        
        # Extract structured information
        key_info = extract_key_information(best_result)
        
        if key_info:
            for key, value in key_info.items():
                if key == 'prices':
                    print(f"💰 Prices found: {value}")
                elif key == 'total_found':
                    print(f"💵 Likely total: ¥{value:,}")
                elif key == 'phone':
                    print(f"📞 Phone: {value}")
                elif key == 'year':
                    print(f"📅 Year: {value}")
                elif key == 'time':
                    print(f"🕐 Time: {value}")
                elif key == 'receipt_numbers':
                    print(f"🧾 Receipt #: {', '.join(value)}")
                elif key == 'percentages':
                    print(f"📊 Percentages: {value}%")
                else:
                    print(f"ℹ️  {key}: {value}")
        else:
            print("❌ No structured information could be extracted")
            
        # Language-specific suggestions
        print("\n" + "=" * 60)
        print("💡 LANGUAGE-SPECIFIC SUGGESTIONS:")
        print("=" * 60)
        print("🇯🇵 For Japanese receipts:")
        print("   • Use 'jpn+eng' for mixed Japanese/English text")
        print("   • Use 'jpn' for purely Japanese text")
        print("   • Ensure good image quality for complex characters")
        print("🌐 Language codes available:")
        print("   • jpn: Japanese")
        print("   • eng: English") 
        print("   • jpn+eng: Combined Japanese and English (recommended)")
        
    else:
        print("❌ Failed to extract text from the receipt image")
        print("💡 Try ensuring the image is clear and well-lit")

if __name__ == "__main__":
    main() 