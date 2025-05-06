# ui/vision_view.py

import streamlit as st
from vision.image_classifier import classify_image
from vision.ocr_processor import extract_text
import os
import random
import re

def process_all_ocr_images():
    ocr_dir = "data/vision"
    output_dir = "data/documents"

    for filename in os.listdir(ocr_dir):
        if filename.endswith("_ocr.png"):
            img_path = os.path.join(ocr_dir, filename)
            text = extract_text(img_path)

            # Extract values using regex
            claim_id_match = re.search(r"Form\s+#?(\d+)", text)
            issue_match = re.search(r"Reported\s+Issue:\s*(.*)", text)
            cost_match = re.search(r"Estimated\s+Cost:\s*\$?([0-9,\.]+)", text)

            if not all([claim_id_match, issue_match, cost_match]):
                print(f"⚠️ Failed to extract all fields from {filename}")
                continue

            claim_id = f"C{int(claim_id_match.group(1)):06d}"
            issue = issue_match.group(1).strip()
            cost = cost_match.group(1).replace(",", "")

            with open(os.path.join(output_dir, f"{claim_id}.txt"), "w") as f:
                f.write(f"claim_id: {claim_id}\n")
                f.write(f"reported_issue: {issue}\n")
                f.write(f"estimated_cost: {cost}\n")

            print(f"✅ Processed and saved: {claim_id}")

def render():
    st.header("🧠 Vision Agent - Damage Classification + OCR")

    image_dir = "data/vision"
    ocr_images = [f for f in os.listdir(image_dir) if f.lower().endswith("_ocr.png")]

    if not ocr_images:
        st.error("❌ No OCR images found.")
        return

    if st.button("📁 Process All OCR Claim Forms"):
        process_all_ocr_images()
        st.success("All OCR claim forms processed and saved.")

    # Pick or select an image
    selected_image = random.choice(ocr_images)
    image_path = os.path.join(image_dir, selected_image)

    st.image(image_path, caption=f"Input Image: {selected_image}", use_container_width=True)

    # Run OCR
    with st.spinner("Running OCR..."):
        try:
            text = extract_text(image_path)
            st.subheader("📝 Extracted Text")
            st.text(text)
        except Exception as e:
            st.error(f"OCR failed: {e}")

    # Run Classification
    with st.spinner("Running Damage Classification..."):
        try:
            predicted_label = classify_image(image_path)
            st.subheader("🔍 Predicted Damage Type")
            st.success(predicted_label)
        except Exception as e:
            st.error(f"Classification failed: {e}")
