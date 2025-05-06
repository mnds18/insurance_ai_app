from PIL import Image, ImageDraw, ImageFont
import os
import random

# Output directory
output_dir = "data/vision_test_images"
os.makedirs(output_dir, exist_ok=True)

# Labels and damage descriptions
labels = ["scratch", "dent", "broken_glass", "fire_damage", "water_damage"]
ocr_templates = [
    "Claim ID: {}\nDamage: {}\nAmount: ${}",
    "Form #{}\nReported Issue: {}\nEstimated Cost: ${}",
    "Incident Ref: {}\nDamage Noted: {}\nRepair Quote: ${}"
]

# Font setup
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

# Function to generate dummy classification image
def create_damage_image(label, index):
    img = Image.new('RGB', (256, 256), color=(255 - index * 2, 255 - index * 5, 255 - index * 3))
    d = ImageDraw.Draw(img)
    d.rectangle([(50, 80), (200, 180)], fill=(200, 0, 0), outline=(0, 0, 0))
    d.text((60, 190), f"{label.replace('_', ' ').title()} {index+1}", fill=(0, 0, 0), font=font)
    file_path = os.path.join(output_dir, f"{label}_{index+1}.jpg")
    img.save(file_path)

# Function to generate dummy OCR form image
def create_ocr_image(label, index):
    ocr_img = Image.new('RGB', (400, 150), color=(255, 255, 255))
    d = ImageDraw.Draw(ocr_img)
    template = random.choice(ocr_templates)
    lines = template.format(1000 + index, label.replace('_', ' ').title(), random.randint(1000, 5000)).split("\n")
    for j, line in enumerate(lines):
        d.text((10, 10 + j * 30), line, fill=(0, 0, 0), font=font)
    ocr_path = os.path.join(output_dir, f"{label}_{index+1}_ocr.png")
    ocr_img.save(ocr_path)

# Create 10 images per class
for label in labels:
    for i in range(10):
        create_damage_image(label, i)
        create_ocr_image(label, i)

output_dir
