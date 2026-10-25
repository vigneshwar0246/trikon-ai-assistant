import asyncio
import json
import pytest
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import OllamaModelMissingError, OllamaUnavailableError
from app.services.pipeline import VoicePipeline
from app.services.session_manager import SessionCapacityError, SessionManager


def test_chat_uses_session_history_and_returns_measured_fields() -> None:
    class FakeOllama:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        async def chat(self, messages: list[dict[str, str]]):
            self.calls.append(messages)
            return f"Reply {len(self.calls)}", {"total_duration": 12_000_000}

    fake = FakeOllama()
    with TestClient(app) as client:
        app.state.sessions = SessionManager(max_turns=3, max_sessions=3)
        app.state.pipeline = VoicePipeline(llm=fake, sessions=app.state.sessions)
        first = client.post("/api/chat", json={"message": "Hello", "session_id": "test-session"})
        second = client.post("/api/chat", json={"message": "Continue", "session_id": "test-session"})
        history = client.get("/api/chat/sessions")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["response"] == "Reply 2"
    assert second.json()["metrics"]["llm_ms"] == 12
    assert fake.calls[1] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Continue"},
    ]
    assert history.json()[0]["message_count"] == 4


def test_chat_stream_emits_real_chunks_and_commits_once_complete() -> None:
    class FakeOllama:
        async def chat_stream(self, messages, *, num_predict=None):
            assert messages[-1] == {"role": "user", "content": "Hello"}
            yield {"token": "Streamed "}
            yield {"token": "answer"}
            yield {"done": True, "metadata": {"total_duration": 9_000_000}}

    sessions = SessionManager(max_turns=3, max_sessions=3)
    with TestClient(app) as client:
        app.state.sessions = sessions
        app.state.pipeline = VoicePipeline(llm=FakeOllama(), sessions=sessions)
        response = client.post("/api/chat/stream", json={"message": "Hello", "session_id": "stream-test"})
        history = client.get("/api/chat/sessions")

    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert [event.get("delta") for event in events if event["type"] == "text"] == ["Streamed ", "answer"]
    assert events[-1]["response"] == "Streamed answer"
    assert events[-1]["metrics"]["first_token_ms"] is not None
    assert history.json()[0]["message_count"] == 2


def test_chat_reports_ollama_offline_and_missing_model() -> None:
    with TestClient(app) as client:
        app.state.pipeline.chat = AsyncMock(side_effect=OllamaUnavailableError("Ollama is offline."))
        offline = client.post("/api/chat", json={"message": "Hi"})
        app.state.pipeline.chat = AsyncMock(side_effect=OllamaModelMissingError("Model missing."))
        missing = client.post("/api/chat", json={"message": "Hi"})

    assert offline.status_code == 503
    assert "offline" in offline.json()["detail"]
    assert missing.status_code == 503
    assert "missing" in missing.json()["detail"]


def test_chat_validates_empty_message_and_can_clear_session() -> None:
    with TestClient(app) as client:
        empty = client.post("/api/chat", json={"message": "   "})
        app.state.sessions = SessionManager()
        asyncio.run(app.state.sessions.get_or_create("clear-me"))
        deleted = client.delete("/api/chat/clear-me")
        deleted_again = client.delete("/api/chat/clear-me")

    assert empty.status_code == 422
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert deleted_again.json()["deleted"] is False


def test_session_capacity_stays_bounded_when_existing_session_is_busy() -> None:
    async def check_capacity() -> None:
        sessions = SessionManager(max_sessions=1)
        busy = await sessions.get_or_create("busy")
        await busy.lock.acquire()
        try:
            with pytest.raises(SessionCapacityError):
                async with sessions.conversation("new"):
                    pass
            assert len(await sessions.list_sessions()) == 1
        finally:
            busy.lock.release()

    asyncio.run(check_capacity())
