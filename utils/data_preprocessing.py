import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dateutil import parser
import re
import os

output_dir = "data"
docs_dir = os.path.join(output_dir, "documents")
images_dir = "images"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def clean_incident_date(date_str):
    try:
        return parser.parse(str(date_str)).date()
    except:
        return np.nan

def normalize_text(text):
    if pd.isnull(text):
        return ""
    # Remove non-alphanumeric characters, lower case
    return re.sub(r'[^A-Za-z0-9\s]', '', text).lower()

def preprocess_claims_data(df, drop_na=True, encode_labels=True):
    print("🧹 Cleaning insurance claims data...")

    # Fix and parse dates
    df["incident_date_clean"] = df["incident_date"].apply(clean_incident_date)

    # Handle missing values
    if drop_na:
        df = df.dropna(subset=["incident_date_clean", "location", "estimated_payout"])

    # Fill or drop NA as needed
    df["claim_text_summary"] = df["claim_text_summary"].fillna("")
    df["claim_text_summary"] = df["claim_text_summary"].apply(normalize_text)

    # Standardize damage_type casing
    df["damage_type"] = df["damage_type"].str.lower().str.strip()

    # Clean name casing
    df["policyholder_name"] = df["policyholder_name"].str.title()

    # Encode categorical variables if requested
    if encode_labels:
        le_loc = LabelEncoder()
        le_damage = LabelEncoder()
        df["location_enc"] = le_loc.fit_transform(df["location"].astype(str))
        df["damage_type_enc"] = le_damage.fit_transform(df["damage_type"].astype(str))

    # Drop unusable rows
    df = df[df["estimated_payout"] > 0]

    # Final column cleanup
    df = df.drop(columns=["incident_date", "location", "damage_type"])
    df = df.rename(columns={"incident_date_clean": "incident_date"})

    print("✅ Preprocessing complete. Records retained:", len(df))
    return df

# Example use
if __name__ == "__main__":
    df_raw = load_data("../data/claims_data.csv")
    df_clean = preprocess_claims_data(df_raw)
    df_clean.to_csv("../data/claims_data_cleaned.csv", index=False)
    print("📁 Cleaned data saved to ../data/claims_data_cleaned.csv")
