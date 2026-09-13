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

## Tech stack

Python · pandas · Faker · scikit-learn · SHAP · matplotlib · FastAPI (planned) ·
LangChain (planned) · FAISS (planned) · PostgreSQL (planned)