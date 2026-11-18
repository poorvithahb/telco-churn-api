# End-to-End MLOps Automation Platform

## 📌 Project Overview
This project is a production-grade Machine Learning Operations (MLOps) pipeline designed to predict Customer Churn for a Telecom company. It goes beyond a standard Jupyter Notebook by automating data preprocessing, model training, experiment tracking, API deployment, and data drift monitoring.

## 🛠️ Tech Stack
- **Language:** Python
- **ML Framework:** Scikit-Learn
- **Experiment Tracking:** MLflow (SQLite backend)
- **API Deployment:** FastAPI & Uvicorn
- **Monitoring:** Evidently AI

## 📂 Project Structure
```text
mlops-project/
├── api/                  # FastAPI deployment code
├── data/                 # Raw dataset (Telco Churn)
├── models/               # Saved best_model.pkl
├── monitoring/           # Drift detection scripts & HTML reports
├── src/                  # Core ML code (data prep, training)
├── main.py               # Orchestrator to run the pipeline
├── requirements.txt      # Dependencies
└── README.md             # Documentation