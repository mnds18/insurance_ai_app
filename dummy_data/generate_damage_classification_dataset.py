# generate_damage_classification_dataset.py

import os
from PIL import Image, ImageDraw, ImageFont
import random

# Define output directory
base_dir = "data/damage_classification_dataset"
labels = ["scratch", "dent", "broken_glass", "fire_damage", "water_damage"]
os.makedirs(base_dir, exist_ok=True)

# Font setup
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

# Generator function for each damage label
def create_image(label, index):
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Add a colored shape
    shape_color = {
        "scratch": (180, 0, 0),
        "dent": (0, 0, 180),
        "broken_glass": (100, 100, 100),
        "fire_damage": (255, 80, 0),
        "water_damage": (0, 100, 255),
    }[label]
    draw.rectangle([60, 60, 160, 160], fill=shape_color)

    # Label
    draw.text((10, 190), f"{label.replace('_', ' ').title()} #{index}", fill=(0, 0, 0), font=font)

    # Save image
    label_dir = os.path.join(base_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    img.save(os.path.join(label_dir, f"{label}_{index}.jpg"))

# Generate 100 images per label (adjust if needed)
for label in labels:
    for i in range(100):
        create_image(label, i)

print("✅ Dataset created at:", base_dir)
