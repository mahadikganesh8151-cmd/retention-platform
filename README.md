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
- [ ] Churn prediction model
- [ ] RAG support chatbot
- [ ] Retention agent
- [ ] Deployment

## Data pipeline

A custom Python scheduler (`pipeline/flow.py`) runs two ordered steps daily:
generate synthetic customer data → engineer features (tenure buckets, average
monthly spend, support-usage flags). Includes logging and failure handling —
if a step fails, downstream steps don't run. Built to be portable to
Airflow/Prefect for production-scale orchestration.

## Tech stack

Python · pandas · Faker · FastAPI (planned) · scikit-learn (planned) ·
LangChain (planned) · FAISS (planned) · PostgreSQL (planned)