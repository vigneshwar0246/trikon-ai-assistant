"""Orchestration for text chat and the full local voice pipeline."""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from app.services.llm import OllamaClient, OllamaResponseError, ollama_client, validate_response
from app.services.language_quality import HINDI_INSTRUCTION
from app.services.commands import Command, route_command
from app.services.diagnostics import collect_diagnostics, status_summary
from app.services.session_manager import SessionManager, session_manager
from app.services.stt import SpeechToTextService, TranscriptionError, stt_service
from app.services.tts import TextToSpeechService, tts_service

VOICE_SYSTEM_PROMPT = (
    "You are a helpful, accurate voice assistant speaking to one person. "
    "Answer directly in one or two short, natural spoken sentences (under 35 words). "
    "Use plain spoken language without markdown, headings, bullets, emoji, code, URLs, or raw symbol characters. "
    "Never read out punctuation or symbol names. Explain technical ideas simply. "
    "Give more detail only when the user asks for it. "
    "If the request is unclear, ask one concise clarifying question."
)

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi"}
ROMAN_HINDI_MARKERS = re.compile(r"\b(?:aap|batao|bharat|hai|hain|hindi|kya|ki|ka|ke|kyun|mujhe|rajdhani|tum|yeh)\b", re.IGNORECASE)


def inferred_text_language(message: str) -> str | None:
    """Identify clear Hindi script or common Roman-Hindi phrasing without guessing all Latin text."""
    if re.search(r"[\u0900-\u097f]", message) or ROMAN_HINDI_MARKERS.search(message):
        return "hi"
    return None


@dataclass
class PipelineResult:
    session_id: str
    transcript: str
    response: str
    audio_path: Path | None
    metrics: dict[str, float | None]
    created_at: datetime
    action: str | None = None
    language: str | None = None
    language_confidence: float | None = None
    speech_available: bool = True


@dataclass
class ChatResult:
    session_id: str
    response: str
    metrics: dict[str, float | None]
    created_at: datetime


class VoicePipeline:
    def __init__(
        self,
        llm: OllamaClient | None = None,
        sessions: SessionManager | None = None,
        stt: SpeechToTextService | None = None,
        tts: TextToSpeechService | None = None,
        voice_num_predict: int = 80,
        voice_hi_num_predict: int = 96,
        hindi_num_predict: int = 192,
        hindi_model: str | None = None,
    ) -> None:
        self.llm = llm or ollama_client
        self.sessions = sessions or session_manager
        self.stt = stt or stt_service
        self.tts = tts or tts_service
        self.voice_num_predict = voice_num_predict
        self.voice_hi_num_predict = voice_hi_num_predict
        self.hindi_num_predict = hindi_num_predict
        self.hindi_model = hindi_model

    def _model_for_language(self, language: str | None) -> str | None:
        return self.hindi_model if language == "hi" else None

    @staticmethod
    def _system_prompt(language: str | None, *, voice: bool = False) -> str:
        base = VOICE_SYSTEM_PROMPT if voice else "You are an accurate local assistant. Answer directly and concisely in one or two short sentences."
        if language == "hi":
            return f"{base}\n\n{HINDI_INSTRUCTION}"
        return (
            f"{base}\n\n"
            "Language: Reply in English unless the user explicitly requests another language.\n"
            "If the user writes in Hindi or Hinglish (Roman Hindi), reply in Hindi using Devanagari.\n"
            "Answer the latest question; use history only for relevant follow-ups.\n"
            "If unsure, say you do not know."
        )

    async def _resolve_command(self, command: Command) -> Command:
        if command.action != "system_status":
            return command
        report = await collect_diagnostics(llm=self.llm, stt=self.stt, tts=self.tts, verify_inference=True)
        return Command(status_summary(report), command.action)

    async def chat(self, message: str, session_id: str | None = None, language: str | None = None) -> ChatResult:
        started = perf_counter()
        async with self.sessions.conversation(session_id) as session:
            if language:
                session.language = language
            elif inferred_text_language(message):
                session.language = "hi"
            messages = self.sessions.history(session)
            if session.language:
                messages.insert(0, {
                    "role": "system",
                    "content": self._system_prompt(session.language),
                })
            command = route_command(message, next((item.content for item in reversed(session.messages) if item.role == "assistant"), None))
            if command:
                response_text = (await self._resolve_command(command)).response
                llm_ms = None
            else:
                messages.append({"role": "user", "content": message})
                model = self._model_for_language(session.language)
                if session.language == "hi":
                    options = {"num_predict": self.hindi_num_predict}
                    if model:
                        options["model"] = model
                    response_text, metadata = await self.llm.chat(messages, **options)
                elif model:
                    response_text, metadata = await self.llm.chat(messages, model=model)
                else:
                    response_text, metadata = await self.llm.chat(messages)
                validate_response(response_text, metadata)
                llm_ms = round((metadata.get("total_duration", 0) or 0) / 1_000_000, 2)
                if llm_ms <= 0:
                    llm_ms = round((perf_counter() - started) * 1000, 2)
            self.sessions.add_exchange(session, message, response_text)
        return ChatResult(
            session_id=session.session_id,
            response=response_text,
            metrics={"llm_ms": llm_ms, "total_ms": round((perf_counter() - started) * 1000, 2)},
            created_at=datetime.now(timezone.utc),
        )

    async def chat_stream(
        self, message: str, session_id: str | None = None, language: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one text turn and commit history only after a complete response."""
        started = perf_counter()
        async with self.sessions.conversation(session_id) as session:
            if language:
                session.language = language
            elif inferred_text_language(message):
                session.language = "hi"
            command = route_command(
                message, next((item.content for item in reversed(session.messages) if item.role == "assistant"), None)
            )
            if command:
                command = await self._resolve_command(command)
                response_text = command.response
                yield {"type": "text", "delta": response_text}
                if command.action == "clear_conversation":
                    session.messages.clear()
                else:
                    self.sessions.add_exchange(session, message, response_text)
                yield {
                    "type": "done", "session_id": session.session_id, "response": response_text,
                    "action": command.action,
                    "metrics": {"llm_ms": None, "first_token_ms": None,
                                "total_ms": round((perf_counter() - started) * 1000, 2)},
                }
                return

            messages = self.sessions.history(session)
            if session.language:
                messages.insert(0, {"role": "system", "content": self._system_prompt(session.language)})
            messages.append({"role": "user", "content": message})
            response_text = ""
            metadata: dict[str, Any] = {}
            first_token_ms: float | None = None
            model = self._model_for_language(session.language)
            if session.language == "hi":
                options = {"num_predict": self.hindi_num_predict}
                if model:
                    options["model"] = model
                stream = self.llm.chat_stream(messages, **options)
            elif model:
                stream = self.llm.chat_stream(messages, model=model)
            else:
                stream = self.llm.chat_stream(messages)
            async for event in stream:
                token = event.get("token")
                if isinstance(token, str) and token:
                    if first_token_ms is None:
                        first_token_ms = round((perf_counter() - started) * 1000, 2)
                    response_text += token
                    yield {"type": "text", "delta": token}
                if event.get("done") and isinstance(event.get("metadata"), dict):
                    metadata = event["metadata"]
            validate_response(response_text, metadata)
            response_text = response_text.strip()
            self.sessions.add_exchange(session, message, response_text)
            duration = metadata.get("total_duration", 0)
            llm_ms = round(duration / 1_000_000, 2) if isinstance(duration, (int, float)) and duration > 0 else round((perf_counter() - started) * 1000, 2)
            yield {
                "type": "done", "session_id": session.session_id, "response": response_text,
                "metrics": {"llm_ms": llm_ms, "first_token_ms": first_token_ms,
                            "total_ms": round((perf_counter() - started) * 1000, 2)},
            }

    async def voice(
        self,
        audio_path: str | Path,
        session_id: str | None = None,
        language: str | None = None,
        *,
        response_language: str | None = None,
    ) -> PipelineResult:
        overall_started = perf_counter()
        async with self.sessions.conversation(session_id) as session:
            stt_started = perf_counter()
            transcript, transcription_details = await asyncio.to_thread(self.stt.transcribe, audio_path, language)
            stt_ms = round((perf_counter() - stt_started) * 1000, 2)
            detected_language = str(transcription_details.get("language") or "")
            confidence_value = transcription_details.get("language_probability")
            language_confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else None
            output_language = response_language or language or detected_language or session.language or "en"
            if output_language not in LANGUAGE_NAMES:
                raise TranscriptionError(
                    "The detected language is not supported for voice output. Select English or Hindi and try again."
                )
            if language is None and language_confidence is not None and language_confidence < self.stt.settings.stt_language_confidence_threshold:
                raise TranscriptionError(
                    f"Language detection was uncertain ({language_confidence:.0%}). Select English or Hindi and try again."
                )
            session.language = output_language
            speech_available = not hasattr(self.tts, "available_for") or self.tts.available_for(output_language)
            last_response = next((item.content for item in reversed(session.messages) if item.role == "assistant"), None)
            command = route_command(transcript, last_response)
            if command:
                command = await self._resolve_command(command)
            if command:
                response_text = command.response
                llm_ms = None
            else:
                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": self._system_prompt(output_language, voice=True),
                    },
                    *self.sessions.history(session),
                ]
                messages.append({"role": "user", "content": transcript})
                llm_started = perf_counter()
                limit = self.voice_hi_num_predict if output_language == "hi" else self.voice_num_predict
                model = self._model_for_language(output_language)
                response_text, metadata = (await self.llm.chat(messages, num_predict=limit, model=model)
                                           if model else await self.llm.chat(messages, num_predict=limit))
                validate_response(response_text, metadata)
                measured_llm_ms = round((perf_counter() - llm_started) * 1000, 2)
                reported_llm_ms = round((metadata.get("total_duration", 0) or 0) / 1_000_000, 2)
                llm_ms = reported_llm_ms if reported_llm_ms > 0 else measured_llm_ms

            audio_output = None
            tts_ms = None
            if speech_available:
                if output_language == "en":
                    audio_output, tts_ms = await asyncio.to_thread(self.tts.synthesize_to_file, response_text)
                else:
                    audio_output, tts_ms = await asyncio.to_thread(self.tts.synthesize_to_file, response_text, output_language)
            if command and command.action == "clear_conversation":
                session.messages.clear()
            else:
                self.sessions.add_exchange(session, transcript, response_text)

        return PipelineResult(
            session_id=session.session_id,
            transcript=transcript,
            response=response_text,
            audio_path=audio_output,
            metrics={
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": round((perf_counter() - overall_started) * 1000, 2),
            },
            created_at=datetime.now(timezone.utc),
            action=command.action if command else None,
            language=output_language,
            language_confidence=language_confidence,
            speech_available=speech_available,
        )

    @staticmethod
    def _take_speakable_chunk(buffer: str, *, final: bool = False, language: str = "en") -> tuple[str | None, str]:
        """Take a sentence/phrase boundary without splitting arbitrary short word groups."""
        if final:
            return (buffer.strip() or None), ""
        matches = list(re.finditer(r"[.!?\u0964\u0965](?:\s+|$)", buffer))
        if matches and matches[-1].end() >= 8:
            end = matches[-1].end()
            return buffer[:end].strip(), buffer[end:]
        # Hindi uses more model tokens per word; start playback before a long sentence finishes.
        limit, phrase_start, word_start = (72, 36, 44) if language == "hi" else (90, 48, 64)
        if len(buffer) >= limit:
            split_at = max(buffer.rfind(", ", phrase_start), buffer.rfind("; ", phrase_start), buffer.rfind(" ", word_start))
            if split_at > 0:
                return buffer[:split_at + 1].strip(), buffer[split_at + 1:]
        return None, buffer

    async def voice_stream(
        self,
        audio_path: str | Path,
        session_id: str | None = None,
        language: str | None = None,
        *,
        response_language: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream text tokens and sentence-safe Piper WAV chunks for one voice turn."""
        overall_started = perf_counter()
        async with self.sessions.conversation(session_id) as session:
            stt_started = perf_counter()
            transcript, details = await asyncio.to_thread(self.stt.transcribe, audio_path, language)
            stt_ms = round((perf_counter() - stt_started) * 1000, 2)
            detected = str(details.get("language") or "")
            probability = details.get("language_probability")
            confidence = float(probability) if isinstance(probability, (int, float)) else None
            output_language = response_language or language or detected or session.language or "en"
            if output_language not in LANGUAGE_NAMES:
                raise TranscriptionError("Select English or Hindi; the detected language is not supported.")
            if language is None and confidence is not None and confidence < self.stt.settings.stt_language_confidence_threshold:
                raise TranscriptionError(
                    f"Language detection was uncertain ({confidence:.0%}). Select English or Hindi and try again."
                )
            session.language = output_language
            speech_available = self.tts.available_for(output_language)
            yield {
                "type": "transcript", "session_id": session.session_id, "transcript": transcript,
                "language": output_language, "language_confidence": confidence, "stt_ms": stt_ms,
            }

            last_response = next((item.content for item in reversed(session.messages) if item.role == "assistant"), None)
            command = route_command(transcript, last_response)
            if command:
                command = await self._resolve_command(command)
                response_text = command.response
                audio_path_out = None
                tts_ms = None
                if speech_available:
                    audio_path_out, tts_ms = await asyncio.to_thread(
                    self.tts.synthesize_to_file, response_text, output_language
                    )
                yield {"type": "text", "delta": response_text}
                if audio_path_out:
                    yield {"type": "audio", "url": f"/api/audio/{audio_path_out.name}", "text": response_text, "index": 0}
                if command.action == "clear_conversation":
                    session.messages.clear()
                else:
                    self.sessions.add_exchange(session, transcript, response_text)
                yield {
                    "type": "done", "session_id": session.session_id, "response": response_text,
                    "action": command.action, "language": output_language, "speech_available": speech_available,
                    "speech_detail": None if speech_available else f"Local {LANGUAGE_NAMES[output_language]} speech output is not installed; the text reply is available.",
                    "metrics": {"stt_ms": stt_ms, "llm_ms": None, "tts_ms": tts_ms,
                                "first_token_ms": None, "first_audio_ms": round((perf_counter() - overall_started) * 1000, 2),
                                "total_ms": round((perf_counter() - overall_started) * 1000, 2)},
                }
                return

            messages: list[dict[str, Any]] = [{
                "role": "system",
                "content": self._system_prompt(output_language, voice=True),
            }, *self.sessions.history(session), {"role": "user", "content": transcript}]
            llm_started = perf_counter()
            first_token_ms: float | None = None
            first_audio_ms: float | None = None
            tts_ms = 0.0
            response_text = ""
            buffer = ""
            audio_index = 0
            reported_llm_ms: float | None = None
            final_metadata: dict[str, Any] = {}
            limit = self.voice_hi_num_predict if output_language == "hi" else self.voice_num_predict
            model = self._model_for_language(output_language)
            stream = (self.llm.chat_stream(messages, num_predict=limit, model=model)
                      if model else self.llm.chat_stream(messages, num_predict=limit))
            async for event in stream:
                token = event.get("token")
                if isinstance(token, str) and token:
                    if first_token_ms is None:
                        first_token_ms = round((perf_counter() - llm_started) * 1000, 2)
                    response_text += token
                    buffer += token
                    # Hindi is buffered until complete-response validation, so a
                    # truncated answer is never shown or spoken as if it were final.
                    if output_language != "hi":
                        yield {"type": "text", "delta": token}
                        chunk, buffer = self._take_speakable_chunk(buffer, language=output_language)
                        if chunk and speech_available:
                            chunk_path, chunk_ms = await asyncio.to_thread(
                                self.tts.synthesize_to_file, chunk, output_language
                            )
                            tts_ms += chunk_ms
                            if first_audio_ms is None:
                                first_audio_ms = round((perf_counter() - overall_started) * 1000, 2)
                            yield {"type": "audio", "url": f"/api/audio/{chunk_path.name}", "text": chunk, "index": audio_index}
                            audio_index += 1
                if event.get("done") and isinstance(event.get("metadata"), dict):
                    final_metadata = event["metadata"]
                    duration_ns = event["metadata"].get("total_duration", 0)
                    if isinstance(duration_ns, (int, float)) and duration_ns > 0:
                        reported_llm_ms = round(duration_ns / 1_000_000, 2)
            validate_response(response_text, final_metadata)
            if output_language == "hi":
                yield {"type": "text", "delta": response_text}
            final_chunk, _ = self._take_speakable_chunk(buffer, final=True)
            if final_chunk and speech_available:
                chunk_path, chunk_ms = await asyncio.to_thread(
                    self.tts.synthesize_to_file, final_chunk, output_language
                )
                tts_ms += chunk_ms
                if first_audio_ms is None:
                    first_audio_ms = round((perf_counter() - overall_started) * 1000, 2)
                yield {"type": "audio", "url": f"/api/audio/{chunk_path.name}", "text": final_chunk, "index": audio_index}
            response_text = response_text.strip()
            if not response_text:
                raise OllamaResponseError("Ollama returned an empty streaming reply.")
            llm_ms = reported_llm_ms or round((perf_counter() - llm_started) * 1000, 2)
            self.sessions.add_exchange(session, transcript, response_text)
            yield {
                "type": "done", "session_id": session.session_id, "response": response_text,
                "action": None, "language": output_language, "speech_available": speech_available,
                "speech_detail": None if speech_available else f"Local {LANGUAGE_NAMES[output_language]} speech output is not installed; the text reply is available.",
                "metrics": {"stt_ms": stt_ms, "llm_ms": llm_ms,
                            "tts_ms": round(tts_ms, 2) if speech_available else None,
                            "first_token_ms": first_token_ms, "first_audio_ms": first_audio_ms,
                            "total_ms": round((perf_counter() - overall_started) * 1000, 2)},
            }


voice_pipeline = VoicePipeline()
