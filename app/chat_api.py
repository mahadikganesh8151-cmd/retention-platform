from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os
from app.customer_store import get_churn_risk
from app.agent import decide_action
from app.rag_service import load_index_and_chunks, retrieve, generate_answer

app = FastAPI(title="Retention Platform - Chat API")

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
if not os.path.exists("app/faiss_index.bin"):
    raise RuntimeError(
        "FAISS index not found. Run 'python app/rag_service.py' first to build it."
    )
faiss_index, chunks = load_index_and_chunks()
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    churn_risk: Optional[dict] = None
    escalated: bool = False
    agent_decision: Optional[dict] = None
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        churn_info = None
        escalated = False

        agent_decision = None

        if request.customer_id:
            churn_info = get_churn_risk(request.customer_id)
            if churn_info:
                agent_decision = decide_action(request.customer_id, churn_info, request.message)
                if agent_decision["decision"] in ["escalate_human", "offer_discount", "send_email"]:
                    escalated = True

        retrieved = retrieve(request.message, faiss_index, chunks, embedding_model, k=3)

        # Adjust tone if this is a high-risk customer
        context_note = ""
        if escalated:
            context_note = (
                "\nIMPORTANT: This customer has a high predicted churn risk. "
                "Be extra attentive, empathetic, and proactive about resolving "
                "their concern. This conversation has been flagged for human "
                "follow-up."
            )

        answer = generate_answer(request.message + context_note, retrieved)
        sources = list(set(r["source"] for r in retrieved))

        if escalated:
            print(f"[ESCALATION LOG] Customer {request.customer_id} flagged high-risk "
                  f"(probability={churn_info['churn_probability']}) during chat.")

        return ChatResponse(
            answer=answer,
            sources=sources,
            churn_risk=churn_info,
            escalated=escalated,
            agent_decision=agent_decision

        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")