from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_high_risk_customer():
    payload = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 171.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] is True  # this profile should predict churn


def test_predict_low_risk_customer():
    payload = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 45.0, "TotalCharges": 2700.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] is False  # this profile should predict no churn


def test_predict_missing_field_returns_422():
    incomplete_payload = {"gender": "Female", "tenure": 5}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422  # FastAPI's validation error code