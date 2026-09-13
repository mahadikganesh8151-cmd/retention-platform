import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
CHUNK_SIZE = 300  # characters per chunk
CHUNK_OVERLAP = 50


def load_documents(directory: str = KNOWLEDGE_BASE_DIR) -> list[dict]:
    documents = []
    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "source": os.path.basename(filepath),
            "text": text
        })
    return documents
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        text_chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk.strip()
            })
    return all_chunks
def build_index(chunks: list[dict], model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings



import json
from dotenv import load_dotenv

load_dotenv()

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_index_and_chunks():
    index = faiss.read_index("app/faiss_index.bin")
    with open("app/chunks_metadata.json", "r") as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(query: str, index, chunks: list[dict], model: SentenceTransformer, k: int = 3) -> list[dict]:
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "distance": float(dist)
        })
    return results


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in retrieved_chunks])

    prompt = f"""You are a helpful customer support assistant for TelcoRetain.
Answer the customer's question using ONLY the context below. If the context
doesn't contain the answer, say you don't have that information and suggest
they contact support directly. Be concise and clear.

Context:
{context}

Customer question: {query}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text
if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading index...")
    index, chunks = load_index_and_chunks()

    test_query = "Can I cancel my contract early and will I be charged?"
    print(f"\nQuery: {test_query}")

    retrieved = retrieve(test_query, index, chunks, model)
    print(f"\nRetrieved {len(retrieved)} chunks:")
    for r in retrieved:
        print(f"  - [{r['source']}] distance={r['distance']:.4f}: {r['text'][:80]}...")

    answer = generate_answer(test_query, retrieved)
    print(f"\nAnswer:\n{answer}")