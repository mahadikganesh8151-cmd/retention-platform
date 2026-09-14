from fastapi.testclient import TestClient
from app.chat_api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_billing_question():
    response = client.post("/chat", json={"message": "What happens if I pay my bill late?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert "billing.txt" in body["sources"]


def test_chat_contract_question():
    response = client.post("/chat", json={"message": "Can I cancel my two-year contract early?"})
    assert response.status_code == 200
    body = response.json()
    assert "contracts.txt" in body["sources"]


def test_chat_out_of_scope_question():
    response = client.post("/chat", json={"message": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    # Should decline gracefully, not hallucinate an answer
    assert "don't have" in body["answer"].lower() or "contact support" in body["answer"].lower()


def test_chat_missing_message_returns_422():
    response = client.post("/chat", json={})
    assert response.status_code == 422