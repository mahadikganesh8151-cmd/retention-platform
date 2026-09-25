from sentence_transformers import SentenceTransformer
from app.rag_service import load_documents, chunk_documents, build_index
import faiss
import json

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading and chunking documents...")
documents = load_documents()
chunks = chunk_documents(documents)
print(f"Created {len(chunks)} chunks from {len(documents)} documents")

print("Building FAISS index...")
index, embeddings = build_index(chunks, model)
print(f"Index built with {index.ntotal} vectors of dimension {embeddings.shape[1]}")

faiss.write_index(index, "app/faiss_index.bin")
with open("app/chunks_metadata.json", "w") as f:
    json.dump(chunks, f, indent=2)
print("Saved index and chunk metadata")