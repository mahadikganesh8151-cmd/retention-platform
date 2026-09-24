# Intelligent Customer Retention Platform

An end-to-end system that predicts customer churn, resolves support queries through 
a retrieval-augmented chatbot, and uses an autonomous agent to reason over both 
signals and recommend retention actions  with full observability into every decision.

## Architecture

Data pipeline → Churn model + RAG chatbot → Retention agent → Action (email / discount / escalate)

(Full architecture diagram to be added in docs/)



## Data pipeline

A custom Python scheduler (`pipeline/flow.py`) runs two ordered steps daily:
generate synthetic customer data → engineer features (tenure buckets, average
monthly spend, support-usage flags). Includes logging and failure handling 
if a step fails, downstream steps don't run. Built to be portable to
Airflow/Prefect for production-scale orchestration.

## Churn prediction API

A FastAPI app (`app/main.py`) serves the trained model:

- **`GET /`** — root route, returns a welcome message with links to docs/health
- **`GET /health`** — basic health check, returns `{"status": "ok"}`
- **`POST /predict`** — accepts a customer's profile (contract type, tenure,
  charges, services, etc.) as JSON, returns a churn prediction and probability

Model is loaded once at app startup (not per-request) for performance.
Incoming data is one-hot encoded and column-aligned via `reindex` to exactly
match the feature structure the model was trained on, avoiding a common
encoding-mismatch bug between training and inference. Errors during
prediction are caught and returned as clean HTTP 500 responses instead of
crashing the app.

**Validated with two contrasting test profiles:**
- New customer, fiber optic, month-to-month contract, high charges →
  predicted **78% churn probability**
- Long-tenure customer, two-year contract, DSL, low charges →
  predicted **3% churn probability**

**Test coverage:** 5 pytest tests covering the health check, root route,
high/low-risk prediction accuracy, and input validation (422 on malformed
requests). Run with:

pytest tests/test_main.py -v


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

## RAG support chatbot

A retrieval-augmented chatbot (`app/rag_service.py`) answers customer support
questions grounded in a company knowledge base, rather than relying on the
LLM's general training knowledge.

**Pipeline:**
1. **Knowledge base** — 4 short policy documents (billing, contracts,
   internet service, cancellation) for a fictional telecom company,
   TelcoRetain
2. **Chunking** — documents split into 300-character chunks with 50-character
   overlap, preserving context across chunk boundaries
3. **Embedding** — chunks embedded with `all-MiniLM-L6-v2`
   (sentence-transformers), consistent with the original DreamConnect project
4. **Indexing** — embeddings stored in a FAISS `IndexFlatL2` index (exact
   nearest-neighbor search — appropriate at this scale; an approximate index
   like IVF/HNSW would only be justified at millions of vectors)
5. **Retrieval** — top-k (k=3) most relevant chunks retrieved per query via
   L2 distance
6. **Generation** — Gemini (`gemini-2.5-flash`, via the current `google-genai`
   SDK) generates an answer, explicitly instructed to use ONLY the retrieved
   context and to say "I don't know" rather than guess — this groundedness
   constraint is what makes it RAG rather than an ungrounded chatbot

**Retrieval evaluation (`app/evaluate_retrieval.py`):** a labeled test set of
8 representative customer questions, each mapped to its known-correct source
document. Measures **Hit Rate @ k** — whether the correct document appears
in the top-k retrieved chunks.

**Result: 100% Hit Rate @ 3 (8/8)**
- Caveat: this validates the mechanism at small scale (12 chunks, 8 test
  queries); a production system would need a larger, more adversarial
  evaluation set, and hit rate would likely drop somewhat as the knowledge
  base and query diversity grow. The evaluation script is reusable and
  designed to scale with the knowledge base.

### Example
**Query:** "Can I cancel my contract early and will I be charged?"

**Retrieved:** 3 chunks from `contracts.txt`

**Answer:** Correctly explained cancellation terms for all three contract
types (month-to-month, one-year, two-year) with accurate fee amounts, fully
grounded in the retrieved content — no hallucinated details.
### Chat API

Served via a FastAPI `/chat` endpoint (`app/chat_api.py`, port 8001):

- **`GET /health`** — health check
- **`POST /chat`** — accepts a customer message, returns a grounded answer
  plus the source documents used

Embedding model and FAISS index loaded once at startup, same pattern as the
churn API. Response includes `sources` so the answer is auditable — you can
verify exactly which documents it was grounded in.

**Test coverage:** 5 pytest tests, including a groundedness check that
confirms the chatbot declines to answer out-of-scope questions (e.g. "What
is the capital of France?") rather than hallucinating an answer. Note: these
tests make live Gemini API calls, so they run noticeably slower (~70s) than
the fully offline churn model tests — a real trade-off between fast unit
tests and thorough end-to-end tests worth managing in a larger system.

Run with:
```
pytest tests/test_chat_api.py -v
```
### System Integration

/chat accepts an optional customer_id, looks up churn risk via the Phase 1/2 model, and if risk >= 0.5 threshold, adjusts the LLM's tone and sets an escalated flag. Include both real test results: customer 7590-VHVEG (80.5% risk, escalated: true) and customer 3655-SNQYZ (1.7% risk, escalated: false).
markdown
- [x] Data pipeline
- [x] Churn prediction model
- [x] RAG support chatbot
- [ ] Retention agent
- [ ] Deployment
## Tech stack

## Tech stack

Python ,pandas , Faker , scikit-learn , SHAP , matplotlib , FastAPI ,
sentence-transformers , FAISS , google-genai (Gemini API) , pytest ,
LangChain (planned) , PostgreSQL (planned)
