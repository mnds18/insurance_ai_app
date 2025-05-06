# 🛡️ Insurance AI App: Enterprise-Grade Agentic AI for Claims Forecasting, Risk Optimization & Decision Support

This project demonstrates a modular, end-to-end AI system for the insurance industry that integrates:

- 🔍 Time series forecasting
- 🧠 ML classification & regression
- 🧾 RAG + LangChain agents
- 🖼️ Computer vision (OCR + damage detection)
- 📊 SHAP explainability
- 🧩 MILP optimization (resource allocation)
- 📈 Streamlit UI + MLflow tracking + Docker automation

> ✅ Built for enterprise realism — every component simulates a real-world insurance workflow (claims triage, underwriting, audit, fraud risk, etc.)

---

## 🔧 Tech Stack

| Domain           | Libraries & Tools                                                                 |
|------------------|-----------------------------------------------------------------------------------|
| Forecasting      | Prophet, LightGBM, XGBoost, SHAP, MLflow                                          |
| ML (Supervised)  | LogisticRegression, Ridge, RandomForest, XGBoost, SHAP                            |
| Optimization     | SciPy `linprog`, PuLP (MILP), pandas                                              |
| LLM Agents       | LangChain, OpenAI (GPT-4), LangChain Tools, FAISS RAG                             |
| Vision / OCR     | PyTorch + torchvision (ResNet18), Tesseract OCR                                   |
| Frontend         | Streamlit, Matplotlib, Altair                                                     |
| Automation       | `run_pipeline.py`, Windows `.bat` script, Docker                                  |

---

## 🗂️ Project Structure

insurance_ai_app/
│
├── data/ # Input/output datasets and dummy data
│ ├── forecast_data_clean.csv
│ ├── classification_data.csv
│ ├── vision_test_images/
│ └── rag_documents/
│
├── forecasting/ # Forecast pipeline (preprocessing + model)
│ ├── ts_preprocessing.py
│ └── train_forecasting_model.py
│
├── supervised/ # Classification + regression models
│ ├── train_classification.py
│ └── train_regression.py
│
├── vision/ # Image classifier + OCR reader
│ ├── image_classifier.py
│ └── ocr_processor.py
│
├── agents/ # LLM-powered agents
│ ├── risk_analyst_agent.py
│ └── claims_investigator_agent.py
│
├── rag/ # RAG index creation and query
│ ├── build_rag_index.py
│ └── query_rag.py
│
├── optimization/ # MILP resource optimization module
│ └── claims_optimizer.py
│
├── notebooks/ # Explainability, EDA, optimization demo
│ ├── forecast_eda.py
│ ├── classification_explainability.py
│ └── optimization_explainer.py
│
├── streamlit_app.py # Full visual UI with agent trigger + forecasts
├── run_pipeline.py # CLI orchestrator for the full pipeline
├── Dockerfile # Docker build script for deployment
├── schedule_task.bat # Windows task automation script
└── README.md # You are here



---

## 🚀 Key Features

### 1. 🔮 Forecasting

- Forecast `total_payout` using Prophet, XGBoost, LightGBM (with cross-validation)
- Feature engineering (lags, rolling stats, holidays)
- MLflow tracking + SHAP explainability + chart saving
- Auto-model selection based on MAPE

### 2. ✅ Supervised ML

- `train_classification.py`: Logistic Regression, RandomForest, XGBoost
- `train_regression.py`: Ridge, RF, XGBoost for payout estimation
- Feature engineering reused from forecasting pipeline
- Full metrics, confusion matrix, SHAP, MLflow logging

### 3. 👁️ Vision + OCR

- Dummy claim damage classification using ResNet18
- OCR form text extraction using Tesseract
- 20+ realistic dummy images auto-generated

### 4. 🧠 Agentic AI (LangChain)

- `risk_analyst_agent.py`: GPT-4 summarizes trends from EDA
- `claims_investigator_agent.py`: Stepwise reasoning with tool-calling (policy check)

### 5. 📄 RAG + LLM

- Upload real PDF policy docs
- Build FAISS index + query with GPT-4
- Simulate customer chatbot / underwriter assistant

### 6. 📦 Optimization (NEW)

- `claims_optimizer.py`: MILP formulation to allocate limited assessors to risky claims
- Uses `scipy.optimize.linprog` or `pulp`
- Visual impact analysis in `optimization_explainer.py`
- Logged via MLflow + CLI runnable

---

## 🖥️ Streamlit UI

Launch with:

```bash
streamlit run streamlit_app.py

```

Includes:

Forecast comparison (model selection, metrics)

Claim classification + regression

Vision OCR & damage prediction

RAG + Risk Summary agents (GPT)

Optimization trigger + summary

🐳 Docker Build
```bash

docker build -t insurance_forecast_app .
docker run -p 8501:8501 insurance_forecast_app

```
⚙️ Daily Automation (Windows Task Scheduler)
Schedule daily run of the full pipeline with:

```bash
schedule_task.bat
```
It auto-triggers:

-Dummy data generation

-Forecast retraining

-Agent runs

-MLflow logging

📊 MLflow UI
To view tracked models, metrics, SHAP plots:

```bash
mlflow ui
# Navigate to http://127.0.0.1:5000

```

✅ Setup & Install
```bash
git clone https://github.com/yourname/insurance_ai_app.git
cd insurance_ai_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Optional
brew install tesseract   # (Mac)
choco install tesseract  # (Windows)

```

🤖 Project Highlights 
Enterprise realism: Mirrors actual insurance workflows (triage, pricing, audit)

Modular: Forecast, CV, OCR, optimization, agents — all integrated

Portable: Streamlit, Docker, MLflow, CLI, .bat automation

Smart: GPT-4 agents invoke reasoning steps and real tools

Demonstrable: Works locally end-to-end on a personal laptop

📬 Questions or Suggestions?
Feel free to connect on LinkedIn or message me through GitHub.

👨‍💻 Built with ❤️ by Mrigendranath Debsarma – AI Architect | GenAI Engineer | Insurance Analytics Specialist


Let me know if you'd like:
- a **PDF version** of this `README`
- a **GitHub `description` + tags**
- a **short version** for your LinkedIn post or GitHub project blurb

