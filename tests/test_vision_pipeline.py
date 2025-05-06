# tests/test_vision_pipeline.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.image_classifier import classify_image
from vision.ocr_processor import extract_text

def test_vision_pipeline():
    image_path = "data/vision_test_images/scratch_3_ocr.png"
    assert os.path.exists(image_path), "Test image not found."

    ocr_text = extract_text(image_path)
    print("🔍 OCR Text:\n", ocr_text)

    label = classify_image(image_path)
    print("✅ Predicted label:", label)

    if label.lower() in ocr_text.lower():
        print("✅ OCR and classifier agree.")
    else:
        print(f"⚠️ Mismatch: OCR says 'scratch', classifier predicted '{label}'")



if __name__ == "__main__":
    test_vision_pipeline()
