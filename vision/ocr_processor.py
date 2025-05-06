# vision/ocr_processor.py

from PIL import Image
import pytesseract

# Explicitly set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path: str) -> str:
    """
    Extract text from a given image path using Tesseract OCR.
    
    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load image: {e}")

    # Tesseract OCR config: OEM 3 = default, PSM 6 = assume single uniform block of text
    custom_config = r'--oem 3 --psm 6'

    try:
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"❌ OCR failed: {e}")

# Optional: Direct run for debugging or CLI use
if __name__ == "__main__":
    import os
    import random

    image_dir = os.path.normpath("data/vision_test_images")
    ocr_images = [f for f in os.listdir(image_dir) if f.lower().endswith("_ocr.png")]

    if not ocr_images:
        raise FileNotFoundError("❌ No OCR images found in the vision_test_images directory.")

    image_file = random.choice(ocr_images)
    image_path = os.path.join(image_dir, image_file)

    print(f"🔍 Using OCR image: {image_path}")
    extracted_text = extract_text(image_path)

    print("📝 Extracted Text:")
    print("--------------------------------------------------")
    print(extracted_text)
    print("--------------------------------------------------")
