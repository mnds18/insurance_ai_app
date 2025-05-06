import os
import random
import pandas as pd
import numpy as np
from faker import Faker
from PIL import Image, ImageDraw
import string

fake = Faker()
output_dir = "data"
docs_dir = os.path.join(output_dir, "documents")
images_dir = "images"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Parameters
NUM_RECORDS = 10000
DAMAGE_TYPES = ['Rear-end Collision', 'Front-end Collision', 'Water Damage', 'Hail Damage', 'Vandalism']
LOCATIONS = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS']
FRAUD_FLAG_PROB = 0.1
MISLABEL_PROB = 0.05
MISSING_VALUE_PROB = 0.08
DUPLICATE_PROB = 0.01

# Helper Functions
def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_claim_row(i):
    damage = random.choice(DAMAGE_TYPES)
    payout = round(np.random.normal(3500, 1200), 2)

    if 'Water' in damage:
        payout += 1200

    fraud = int(random.random() < FRAUD_FLAG_PROB)

    # Inject mislabels
    if random.random() < MISLABEL_PROB:
        fraud = 1 - fraud  # Flip the label

    # Inject payout outliers
    if random.random() < 0.02:
        payout *= random.choice([0.1, 15])  # Extremely low or high

    # Inject missing values
    location = random.choice(LOCATIONS) if random.random() > MISSING_VALUE_PROB else None
    payout = payout if random.random() > MISSING_VALUE_PROB else None

    # Format variation in date
    date_format = random.choice(['%Y-%m-%d', '%d/%m/%Y', '%b %d, %Y'])
    incident_date = fake.date_between(start_date='-2y', end_date='today').strftime(date_format)

    # Inject garbled text
    summary = fake.text(max_nb_chars=180)
    if random.random() < 0.05:
        summary += ' @#*!%' + random_string(4)

    return {
        "claim_id": f"C{i:06d}",
        "policyholder_name": fake.name().upper() if random.random() < 0.2 else fake.name().title(),
        "policy_id": f"P{random.randint(10000,99999)}",
        "location": location,
        "incident_date": incident_date,
        "damage_type": damage,
        "claim_text_summary": summary,
        "estimated_payout": payout,
        "is_fraud": fraud
    }

def generate_pdf(claim_id, summary, is_blank=False):
    file_path = os.path.join(docs_dir, f"{claim_id}.txt")
    with open(file_path, "w") as f:
        if is_blank:
            f.write("")
        else:
            f.write(f"CLAIM ID: {claim_id}\n\n{summary}\n\nAuto-generated doc")
    return file_path

def generate_damage_image(claim_id, is_blank=False):
    image_path = os.path.join(images_dir, f"{claim_id}.jpg")
    if is_blank:
        img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    else:
        img = Image.new("RGB", (128, 128), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        draw = ImageDraw.Draw(img)
        draw.text((10, 50), claim_id, fill=(0, 0, 0))
    img.save(image_path)
    return image_path

# Main generation loop
print(f"🔁 Generating {NUM_RECORDS} synthetic claims with realistic issues...")

claims = []
for i in range(NUM_RECORDS):
    row = generate_claim_row(i)
    is_blank = random.random() < 0.03
    generate_pdf(row["claim_id"], row["claim_text_summary"], is_blank)
    generate_damage_image(row["claim_id"], is_blank)
    claims.append(row)

# Inject some duplicates
duplicates = random.sample(claims, int(DUPLICATE_PROB * NUM_RECORDS))
claims += duplicates

df = pd.DataFrame(claims)
csv_path = os.path.join(output_dir, "claims_data.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Data saved to {csv_path}")
print(f"📄 {len(os.listdir(docs_dir))} claim documents in {docs_dir}")
print(f"🖼️  {len(os.listdir(images_dir))} images in {images_dir}")
