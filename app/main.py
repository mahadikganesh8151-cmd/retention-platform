from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Retention Platform API")

MODEL_PATH = "app/churn_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. Run 'python app/train_model.py' first."
    )

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Retention Platform API is running",
        "docs": "/docs",
        "health": "/health"
    }
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float



@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        input_df = pd.DataFrame([customer.dict()])
        input_df = pd.get_dummies(input_df)

        model_columns = model.named_steps["scaler"].feature_names_in_
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        churn_probability = model.predict_proba(input_df)[0][1]
        churn_prediction = bool(model.predict(input_df)[0])

        return {
            "churn_prediction": churn_prediction,
            "churn_probability": round(float(churn_probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")