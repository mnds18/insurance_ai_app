import subprocess
import os

MODULES = [
    ("Generate Dummy Data", "python dummy_data/generate_dummy_data.py"),
    ("Generate Forecast Data", "python dummy_data/generate_forecast_data_dirty.py"),
    ("Preprocess Time Series", "python forecasting/ts_preprocessing.py"),
    ("Train Forecasting Model", "python forecasting/train_forecasting_model.py"),
    ("Train Classification Model", "python ml/train_classification.py"),
    ("Train Regression Model", "python ml/train_regression.py"),
    ("Train Unsupervised Model", "python ml/train_unsupervised.py"),
    ("Build RAG Index", "python rag/build_rag_index.py"),
    ("Run Notebooks", "python notebooks/run_all.py"),
    ("Run Risk Analyst Agent", "python agents/risk_analyst_agent.py"),
    ("Run Claims Investigator Agent", "python agents/claims_investigator_agent.py")
    ("Run Image Classifier", "python vision/image_classifier.py")
    ("Run OCR Processor", "python vision/ocr_processor.py")
]

def run_all():
    for label, command in MODULES:
        print(f"\n🚀 Running: {label}...")
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"✅ {label} completed.")
        except subprocess.CalledProcessError as e:
            print(f"❌ {label} failed with error: {e}")

if __name__ == "__main__":
    print("\n==============================")
    print("🔁 INSURANCE AI - FULL PIPELINE")
    print("==============================")
    run_all()
    print("\n✅ All modules executed.")
