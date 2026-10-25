"""Async client for the local Ollama HTTP API."""

import logging
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.services.language_quality import text_quality_error

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaModelMissingError(RuntimeError):
    pass


class OllamaResponseError(RuntimeError):
    pass


def validate_response(text: str, metadata: dict[str, Any]) -> None:
    problem = text_quality_error(text)
    if metadata.get("done_reason") == "length":
        problem = "reply exceeded its generation limit"
    if problem:
        raise OllamaResponseError(
            f"The local model could not produce a complete, usable answer ({problem}). "
            "Please rephrase or ask a shorter question."
        )


class OllamaClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.settings.ollama_timeout_seconds))
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def warm_model(self, model: str | None = None) -> None:
        """Keep a local model resident without generating a user-visible reply."""
        selected_model = model or self.settings.ollama_model
        try:
            response = await self._get_client().post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": selected_model, "prompt": "Ready", "stream": False,
                    "keep_alive": "30m", "options": {
                        "num_predict": 1, "num_ctx": self.settings.ollama_num_ctx,
                        "temperature": self.settings.ollama_temperature,
                    },
                },
                timeout=self.settings.ollama_timeout_seconds,
            )
            if response.status_code == 404:
                raise OllamaModelMissingError(
                    f"Ollama model '{selected_model}' is missing. Pull it with 'ollama pull {selected_model}'."
                )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError("Ollama model warm-up timed out.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError("Ollama model warm-up failed.") from exc

    async def list_models(self, timeout_seconds: float | None = None) -> list[str]:
        try:
            response = await self._get_client().get(
                f"{self.settings.ollama_base_url}/api/tags",
                timeout=timeout_seconds or self.settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaUnavailableError("Ollama is not reachable at the configured local URL.") from exc
        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama returned an invalid model list.")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama returned an invalid model list.")
        return [str(item.get("name", "")) for item in models if isinstance(item, dict)]

    async def model_available(self) -> bool:
        names = await self.list_models()
        expected = self.settings.ollama_model
        normalized_names = {name if ":" in name else f"{name}:latest" for name in names}
        normalized_expected = expected if ":" in expected else f"{expected}:latest"
        return normalized_expected in normalized_names

    async def loaded_models(self) -> list[dict[str, int | str]]:
        try:
            response = await self._get_client().get(f"{self.settings.ollama_base_url}/api/ps", timeout=3.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [
                {"name": str(item.get("name", "")), "size_vram_bytes": int(item.get("size_vram", 0))}
                for item in models if isinstance(item, dict)
            ]
        except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
            raise OllamaUnavailableError("Ollama runtime details are unavailable.") from exc

    async def chat(self, messages: list[dict[str, str]], *, num_predict: int | None = None,
                   model: str | None = None) -> tuple[str, dict[str, Any]]:
        """Generate a reply without pulling a missing model automatically."""
        try:
            response = await self._get_client().post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": model or self.settings.ollama_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.settings.ollama_temperature,
                            "num_ctx": self.settings.ollama_num_ctx,
                            "num_predict": num_predict if num_predict is not None else self.settings.ollama_num_predict,
                        },
                        "keep_alive": "15m",
                    },
            )
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError("Ollama did not respond before the request timed out.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError("Ollama is not reachable at the configured local URL.") from exc

        if response.status_code == 404:
            raise OllamaModelMissingError(
                f"Ollama model '{model or self.settings.ollama_model}' is missing. Pull it with 'ollama pull {model or self.settings.ollama_model}'."
            )
        if response.is_error:
            detail = response.text[:300]
            if "model" in detail.lower() and ("not found" in detail.lower() or "pull" in detail.lower()):
                raise OllamaModelMissingError(
                    f"Ollama model '{model or self.settings.ollama_model}' is missing. Pull it with 'ollama pull {model or self.settings.ollama_model}'."
                )
            raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}.")

        try:
            payload = response.json()
            text = payload["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaResponseError("Ollama returned a malformed chat response.") from exc
        if not isinstance(text, str) or not text.strip():
            raise OllamaResponseError("Ollama returned an empty reply.")
        return text.strip(), payload

    async def chat_stream(
        self, messages: list[dict[str, str]], *, num_predict: int | None = None, model: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield Ollama token events and one final metadata event from its NDJSON stream."""
        completed = False
        try:
            async with self._get_client().stream(
                "POST",
                f"{self.settings.ollama_base_url}/api/chat",
                json={
                    "model": model or self.settings.ollama_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.settings.ollama_temperature,
                        "num_ctx": self.settings.ollama_num_ctx,
                        "num_predict": num_predict if num_predict is not None else self.settings.ollama_num_predict,
                    },
                    "keep_alive": "15m",
                },
            ) as response:
                if response.status_code == 404:
                    raise OllamaModelMissingError(
                        f"Ollama model '{model or self.settings.ollama_model}' is missing. Pull it with 'ollama pull {model or self.settings.ollama_model}'."
                    )
                if response.is_error:
                    await response.aread()
                    raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}.")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except ValueError as exc:
                        raise OllamaResponseError("Ollama returned malformed streaming data.") from exc
                    if not isinstance(payload, dict):
                        raise OllamaResponseError("Ollama returned malformed streaming data.")
                    if payload.get("error"):
                        raise OllamaResponseError("Ollama reported a generation error. Please try again.")
                    message = payload.get("message", {})
                    if not isinstance(message, dict):
                        raise OllamaResponseError("Ollama returned a malformed streaming message.")
                    token = message.get("content", "")
                    if not isinstance(token, str):
                        raise OllamaResponseError("Ollama returned non-text content.")
                    if isinstance(token, str) and token:
                        yield {"token": token}
                    if payload.get("done"):
                        completed = True
                        yield {"done": True, "metadata": payload}
                        break
                if not completed:
                    raise OllamaResponseError("Ollama stopped before completing its answer. Please try again.")
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError("Ollama did not respond before the request timed out.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError("Ollama is not reachable at the configured local URL.") from exc


ollama_client = OllamaClient()
