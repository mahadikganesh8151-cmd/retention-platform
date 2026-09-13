# Intelligent Customer Retention Platform

An end-to-end system that predicts customer churn, resolves support queries through 
a retrieval-augmented chatbot, and uses an autonomous agent to reason over both 
signals and recommend retention actions — with full observability into every decision.

## Architecture

Data pipeline → Churn model + RAG chatbot → Retention agent → Action (email / discount / escalate)

(Full architecture diagram to be added in docs/)

## Status

In active development — see branches for work in progress.

- [x] Data pipeline
- [x] Churn prediction model
- [ ] RAG support chatbot
- [ ] Retention agent
- [ ] Deployment

## Data pipeline

A custom Python scheduler (`pipeline/flow.py`) runs two ordered steps daily:
generate synthetic customer data → engineer features (tenure buckets, average
monthly spend, support-usage flags). Includes logging and failure handling —
if a step fails, downstream steps don't run. Built to be portable to
Airflow/Prefect for production-scale orchestration.

## Churn prediction model

Trained on the real IBM Telco Customer Churn dataset (7,043 customers, 21
features). Compared Logistic Regression and Random Forest with 5-fold
cross-validation, optimizing for ROC-AUC due to class imbalance (26.5% churn
rate). Logistic Regression was selected as the production model:

- **Test ROC-AUC: 0.84**
- **Churn recall: 0.79** (after applying `class_weight="balanced"` to
  prioritize catching actual churners over raw accuracy — a deliberate
  precision/recall trade-off, since missing a churner is costlier than a
  false alarm)
- Features scaled with `StandardScaler` inside a scikit-learn `Pipeline`
  (required for Logistic Regression; not needed for the tree-based model)

**Explainability (SHAP):** tenure, monthly charges, and contract type are the
strongest churn drivers. New customers, high monthly bills, fiber optic
service, and month-to-month contracts all push predictions toward churn;
long tenure and two-year contracts push away from it — consistent with
real-world telecom churn behavior. See `reports_shap_importance.png` and
`reports_shap_beeswarm.png`.

## Churn prediction API

A FastAPI app (`app/main.py`) serves the trained model:

- **`GET /health`** — basic health check, returns `{"status": "ok"}`
- **`POST /predict`** — accepts a customer's profile (contract type, tenure,
  charges, services, etc.) as JSON, returns a churn prediction and probability

Model is loaded once at app startup (not per-request) for performance.
Incoming data is one-hot encoded and column-aligned via `reindex` to exactly
match the feature structure the model was trained on, avoiding a common
encoding-mismatch bug between training and inference.

**Validated with two contrasting test profiles:**
- New customer, fiber optic, month-to-month contract, high charges →
  predicted **78% churn probability**
- Long-tenure customer, two-year contract, DSL, low charges →
  predicted **3% churn probability**

Interactive API docs available at `/docs` (FastAPI auto-generated).

### Example request

POST /predict
{
"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
"tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
"InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
"DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
"StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
"PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 171.0
}

### Example response
```json
{
  "churn_prediction": true,
  "churn_probability": 0.7811
}
```

## Tech stack

Python · pandas · Faker · scikit-learn · SHAP · matplotlib · FastAPI (planned) ·
LangChain (planned) · FAISS (planned) · PostgreSQL (planned)