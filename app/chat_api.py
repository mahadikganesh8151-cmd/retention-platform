from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

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
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        retrieved = retrieve(request.message, faiss_index, chunks, embedding_model, k=3)
        answer = generate_answer(request.message, retrieved)
        sources = list(set(r["source"] for r in retrieved))

        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")