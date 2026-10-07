# 🛡️ ChurnGuard AI

**ChurnGuard AI** is an enterprise-grade customer retention platform that leverages Machine Learning (XGBoost) and SHAP explainability to predict customer churn risk, explain key risk drivers, and recommend automated retention playbooks.

---

## 📌 Features

- **XGBoost Machine Learning Pipeline:** Trained on key customer churn features with automated data preprocessing.
- **SHAP Explainability:** Highlights individual feature impacts driving higher or lower churn probability.
- **FastAPI Real-Time Backend:** RESTful endpoint for low-latency batch and single-customer churn risk scoring.
- **Streamlit Interactive UI:** Dashboard featuring single customer assessment, batch CSV risk analysis, and automated retention action playbooks.

---

## 📁 Repository Structure

```text
churn-prediction-system/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── main.py
└── app.py
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/churn-prediction-system.git
cd churn-prediction-system
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Train the Model & Generate Artifacts
Run `train.py` to train the XGBoost model and save all joblib artifacts:
```bash
python train.py
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

### 5. Run the FastAPI Backend (Optional)
```bash
uvicorn main:app --reload --port 8000
```
Access the interactive API docs at `http://127.0.0.1:8000/docs`.

---

## 🛡️ License
MIT License
