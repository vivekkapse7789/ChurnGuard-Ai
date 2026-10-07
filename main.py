from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

app = FastAPI(
    title="ChurnGuard AI API",
    description="Real-time churn risk prediction & explainability service",
    version="1.0.0"
)

preprocessor = joblib.load('preprocessor.joblib')
model = joblib.load('model.joblib')
explainer = joblib.load('explainer.joblib')
feature_names = joblib.load('feature_names.joblib')

class CustomerData(BaseModel):
    Tenure: int = Field(..., ge=0, example=12)
    MonthlyCharges: float = Field(..., ge=0.0, example=75.50)
    TotalCharges: float = Field(..., ge=0.0, example=906.00)
    Contract: str = Field(..., example="Month-to-month")
    InternetService: str = Field(..., example="Fiber optic")
    PaymentMethod: str = Field(..., example="Electronic check")

@app.get("/")
def health_check():
    return {"status": "active", "system": "ChurnGuard AI Backend"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        input_data = pd.DataFrame([customer.dict()])
        processed_data = preprocessor.transform(input_data)
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
        
        churn_prob = float(model.predict_proba(processed_df)[0][1])
        prediction = int(churn_prob >= 0.5)
        
        shap_values = explainer.shap_values(processed_df)[0]
        feature_importance = dict(zip(feature_names, [float(v) for v in shap_values]))
        
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        )

        risk_level = "High" if churn_prob >= 0.65 else ("Medium" if churn_prob >= 0.35 else "Low")

        return {
            "churn_prediction": prediction,
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_level,
            "top_risk_drivers": sorted_importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
