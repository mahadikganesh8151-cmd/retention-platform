from sentence_transformers import SentenceTransformer
from rag_service import load_index_and_chunks, retrieve

# Each test case: a realistic question + which source file SHOULD be retrieved
TEST_CASES = [
    {"query": "Can I cancel my contract early and will I be charged?", "expected_source": "contracts.txt"},
    {"query": "What payment methods do you accept?", "expected_source": "billing.txt"},
    {"query": "Why is my internet so slow?", "expected_source": "internet_service.txt"},
    {"query": "How do I return my router after cancelling?", "expected_source": "cancellation.txt"},
    {"query": "Do you offer any deals if I try to cancel my service?", "expected_source": "cancellation.txt"},
    {"query": "What's the difference between fiber and DSL?", "expected_source": "internet_service.txt"},
    {"query": "What happens if I pay my bill late?", "expected_source": "billing.txt"},
    {"query": "Can I switch from month-to-month to a two-year plan?", "expected_source": "contracts.txt"},
]
def evaluate_retrieval(k: int = 3):
    print("Loading embedding model and index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()

    correct = 0
    results = []

    for case in TEST_CASES:
        retrieved = retrieve(case["query"], index, chunks, model, k=k)
        retrieved_sources = [r["source"] for r in retrieved]
        hit = case["expected_source"] in retrieved_sources

        results.append({
            "query": case["query"],
            "expected": case["expected_source"],
            "retrieved_sources": retrieved_sources,
            "hit": hit
        })

        if hit:
            correct += 1

    hit_rate = correct / len(TEST_CASES)

    print(f"\n=== Retrieval Evaluation (k={k}) ===")
    for r in results:
        status = "PASS" if r["hit"] else "FAIL"
        print(f"[{status}] Q: {r['query']}")
        print(f"       Expected: {r['expected']} | Retrieved: {r['retrieved_sources']}")

    print(f"\nHit Rate @ {k}: {hit_rate:.2%} ({correct}/{len(TEST_CASES)})")
    return hit_rate


if __name__ == "__main__":
    evaluate_retrieval(k=3)