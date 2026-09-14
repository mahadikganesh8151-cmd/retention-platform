import pandas as pd
from typing import Optional
_customer_cache = None


def load_customers() -> pd.DataFrame:
    global _customer_cache
    if _customer_cache is None:
        df = pd.read_csv("data/telco_churn_raw.csv")
        _customer_cache = df
    return _customer_cache

def get_customer_by_id(customer_id: str) -> Optional[dict]:
    df = load_customers()
    match = df[df["customerID"] == customer_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()
import joblib
import pandas as pd

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load("app/churn_model.pkl")
    return _model_cache


def get_churn_risk(customer_id: str) -> Optional[dict]:
    customer = get_customer_by_id(customer_id)
    if customer is None:
        return None

    model = load_model()

    # Build input matching what the model expects, same as the /predict endpoint
    input_data = {k: v for k, v in customer.items() if k not in ["customerID", "Churn"]}
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    model_columns = model.named_steps["scaler"].feature_names_in_
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    churn_probability = model.predict_proba(input_df)[0][1]

    return {
        "customer_id": customer_id,
        "churn_probability": round(float(churn_probability), 4),
        "risk_level": "high" if churn_probability >= 0.5 else "low"
    }
if __name__ == "__main__":
    df = load_customers()

    # Find a customer with long tenure and a two-year contract — likely low risk
    low_risk_candidates = df[(df["tenure"] > 50) & (df["Contract"] == "Two year")]
    sample_id = low_risk_candidates.iloc[0]["customerID"]

    print(f"Testing with likely low-risk customer ID: {sample_id}")
    print(get_churn_risk(sample_id))