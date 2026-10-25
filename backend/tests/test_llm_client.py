import json

import httpx
import pytest

from app.core.config import Settings
from app.services.llm import OllamaClient, OllamaModelMissingError, OllamaResponseError


@pytest.mark.asyncio
async def test_ollama_connection_reuse_and_generation_limit() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "real upstream text"}, "total_duration": 10_000_000})

    client = OllamaClient(Settings(ollama_num_predict=96))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    connection = client._client
    try:
        assert (await client.chat([{"role": "user", "content": "Hello"}]))[0] == "real upstream text"
        assert (await client.chat([{"role": "user", "content": "Again"}]))[0] == "real upstream text"
        assert (await client.chat([{"role": "user", "content": "Voice"}], num_predict=64))[0] == "real upstream text"
        assert client._client is connection
        assert [item["options"]["num_predict"] for item in requests] == [96, 96, 64]
        assert all(item["keep_alive"] == "15m" for item in requests)
    finally:
        await client.close()
    assert connection.is_closed


@pytest.mark.asyncio
async def test_ollama_missing_model_is_reported_without_fabrication() -> None:
    client = OllamaClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    try:
        with pytest.raises(OllamaModelMissingError):
            await client.chat([{"role": "user", "content": "Hello"}])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_warm_model_uses_the_same_runtime_options_as_chat() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    client = OllamaClient(Settings(ollama_model="qwen2.5:3b", ollama_num_ctx=2048, ollama_temperature=0.1))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        await client.warm_model()
    finally:
        await client.close()

    assert requests == [{
        "model": "qwen2.5:3b", "prompt": "Ready", "stream": False, "keep_alive": "30m",
        "options": {"num_predict": 1, "num_ctx": 2048, "temperature": 0.1},
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    [{"error": "generation failed"}],
    [{"message": None}],
    [{"message": {"content": "unfinished"}}],
    [{"message": {"content": 42}, "done": True}],
])
async def test_stream_errors_and_incomplete_streams_are_not_success(payload):
    client = OllamaClient()
    body = "\n".join(json.dumps(event) for event in payload)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)))
    try:
        with pytest.raises(OllamaResponseError):
            _ = [event async for event in client.chat_stream([{"role": "user", "content": "Hello"}])]
    finally:
        await client.close()
