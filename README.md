# Intelligent Customer Retention Platform

An end-to-end system that predicts customer churn, resolves support queries through 
a retrieval-augmented chatbot, and uses an autonomous agent to reason over both 
signals and recommend retention actions with full observability into every decision.

## Architecture

Data pipeline → Churn model + RAG chatbot → Retention agent → Action (email / discount / escalate)

(Full architecture diagram to be added in docs/)

## Status

In active development — see branches for work in progress.

-  Data pipeline (Airflow)
-  Churn prediction model
-  RAG support chatbot
-  Retention agent
-  Deployment

## Tech stack

Python · FastAPI · scikit-learn · LangChain · FAISS · PostgreSQL · Airflow · Docker