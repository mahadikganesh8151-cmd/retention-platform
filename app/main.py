from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Retention Platform API")

model = joblib.load("app/churn_model.pkl")
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
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerInput):
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