import json
import os
import tempfile
import time
from types import SimpleNamespace

import pytest

fd, path = tempfile.mkstemp(suffix=".db")
os.close(fd)
os.environ["DATABASE_URL"] = f"sqlite:///{path}"
os.environ["LLM_API_KEYS"] = "test-key-1,test-key-2"
os.environ["ENABLE_FAILURE_SIMULATION"] = "true"

from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def fake_provider(monkeypatch):
    cache = {}
    calls = []

    def completion(model, messages, api_key, caching):
        assert caching is True
        cache_key = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        if cache_key in cache:
            result = cache[cache_key]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=result[0]))],
                usage=SimpleNamespace(prompt_tokens=result[1], completion_tokens=result[2]),
                _hidden_params={"cache_hit": True},
            )
        calls.append({"model": model, "messages": messages, "api_key": api_key})
        time.sleep(0.01)
        reply = f"Model response to: {messages[-1]['content']}"
        usage = (24 + len(messages), 5)
        cache[cache_key] = (reply, *usage)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
            _hidden_params={"cache_hit": False},
        )

    monkeypatch.setattr(main.litellm, "completion", completion)
    return calls


def test_health_and_live_chat(client, fake_provider):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}

    response = client.post(
        "/chat",
        json={"session_id": "test-live", "message": "What is the capital of France?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "Model response to: What is the capital of France?"
    assert payload["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert payload["metadata"]["key_id"] == "key_1"
    assert payload["metadata"]["input_tokens"] == 26
    assert payload["metadata"]["output_tokens"] == 5


def test_litellm_cache_reused_across_sessions(client, fake_provider):
    body = {"message": "Repeatable cache prompt"}
    first = client.post("/chat", json={"session_id": "cache-a", **body}).json()
    second = client.post("/chat", json={"session_id": "cache-b", **body}).json()

    assert first["reply"] == second["reply"]
    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0
    assert second["metadata"]["input_tokens"] == first["metadata"]["input_tokens"]
    assert second["metadata"]["output_tokens"] == first["metadata"]["output_tokens"]
    assert second["metadata"]["latency_ms"] < first["metadata"]["latency_ms"]
    assert len(fake_provider) == 1


def test_different_prompt_does_not_reuse_litellm_cache(client, fake_provider):
    first = client.post("/chat", json={"session_id": "cache-different-a", "message": "Prompt A"}).json()
    second = client.post("/chat", json={"session_id": "cache-different-b", "message": "Prompt B"}).json()

    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert first["reply"] != second["reply"]
    assert len(fake_provider) == 2


def test_failover_and_history(client, fake_provider):
    response = client.post(
        "/chat",
        json={"session_id": "test-failover", "message": "Use the second key"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["key_id"] == "key_2"
    assert metadata["fallback_attempts"] == 1
    assert fake_provider[-1]["api_key"] == "test-key-2"

    history = client.get("/sessions/test-failover/messages")
    assert history.status_code == 200
    assert [message["role"] for message in history.json()["messages"]] == ["user", "assistant"]


@pytest.mark.parametrize("status_code", [429, 500, 502, 503])
def test_provider_failure_automatically_uses_next_key(client, monkeypatch, status_code):
    attempted_keys = []

    class ProviderError(Exception):
        pass

    def completion(model, messages, api_key, caching):
        attempted_keys.append(api_key)
        if api_key == "test-key-1":
            error = ProviderError()
            error.status_code = status_code
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered on key two"))],
            usage=SimpleNamespace(prompt_tokens=31, completion_tokens=4),
            _hidden_params={"cache_hit": False},
        )

    monkeypatch.setattr(main.litellm, "completion", completion)
    response = client.post(
        "/chat",
        json={"session_id": f"test-provider-fallback-{status_code}", "message": "Try each provider"},
    )

    assert response.status_code == 200
    assert attempted_keys == ["test-key-1", "test-key-2"]
    assert response.json()["metadata"]["key_id"] == "key_2"
    assert response.json()["metadata"]["fallback_attempts"] == 1


def test_multi_turn_history_and_system_prompt(client, fake_provider):
    first = client.post(
        "/chat",
        json={"session_id": "test-turns", "message": "Remember the word orbit", "system_prompt": "Be concise"},
    )
    second = client.post("/chat", json={"session_id": "test-turns", "message": "What word should you remember?"})

    assert first.status_code == second.status_code == 200
    assert fake_provider[-1]["messages"][0] == {"role": "system", "content": "Be concise"}
    assert fake_provider[-1]["messages"][1]["content"] == "Remember the word orbit"
    history = client.get("/sessions/test-turns/messages").json()["messages"]
    assert len(history) == 4
    assert all("metadata" not in message or "model" not in message["metadata"] for message in history)


def test_all_keys_failed_does_not_save_history(client):
    response = client.post(
        "/chat",
        json={"session_id": "test-unavailable", "message": "Never saved"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/test-unavailable/messages").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": " ", "message": "hello"},
        {"session_id": "valid", "message": "   "},
        {"session_id": "valid", "message": "x" * 4001},
    ],
)
def test_invalid_chat_requests_return_422_without_history(client, body):
    response = client.post("/chat", json=body)
    assert response.status_code == 422
    if body["session_id"] == "valid":
        assert client.get("/sessions/valid/messages").status_code == 404


def test_message_length_is_checked_after_trimming(client, fake_provider):
    response = client.post(
        "/chat",
        json={"session_id": "test-trim", "message": f"{' ' * 4001}x{' ' * 4001}"},
    )
    assert response.status_code == 200
    assert response.json()["reply"] == "Model response to: x"
