import os
import tempfile

fd, path = tempfile.mkstemp(suffix=".db")
os.close(fd)
os.environ["DATABASE_URL"] = f"sqlite:///{path}"
os.environ["LLM_API_KEYS"] = "placeholder-1,placeholder-2"
os.environ["ENABLE_FAILURE_SIMULATION"] = "true"

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_and_chat():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    first = client.post("/chat", json={"session_id": "test-1", "message": "What is the capital of France?"})
    assert first.status_code == 200
    assert first.json()["reply"] == "The capital of France is Paris."
    assert first.json()["metadata"]["resolution_source"] in {"LIVE_INFERENCE", "LLM_CACHE"}


def test_failover_and_history():
    response = client.post(
        "/chat",
        json={"session_id": "test-2", "message": "Hello"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["key_id"] == "key_2"
    assert response.json()["metadata"]["fallback_attempts"] == 1

    history = client.get("/sessions/test-2/messages")
    assert history.status_code == 200
    assert len(history.json()["messages"]) == 2


def test_all_keys_failed():
    response = client.post(
        "/chat",
        json={"session_id": "test-3", "message": "Never saved"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/test-3/messages").status_code == 404
