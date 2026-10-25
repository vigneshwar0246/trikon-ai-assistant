"""Audio upload and voice-to-voice pipeline route."""

import logging
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.schemas.chat import MetricsResponse
from app.schemas.voice import VoiceResponse
from app.services.llm import OllamaModelMissingError, OllamaResponseError, OllamaUnavailableError
from app.services.stt import SpeechModelNotInstalledError, TranscriptionError
from app.services.session_manager import SessionCapacityError
from app.services.tts import SpeechSynthesisError, VoiceModelNotInstalledError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".mp4", ".flac", ".mpeg", ".mpga"}
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3", "audio/mp4",
    "audio/x-m4a", "audio/ogg", "audio/webm", "audio/flac", "audio/x-flac", "video/webm",
    "application/octet-stream",
}


async def _save_upload(upload: UploadFile, max_bytes: int) -> Path:
    suffix = Path(upload.filename or "recording.webm").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported audio file. Upload WAV, MP3, M4A, OGG, WebM, or FLAC.")
    content_type = (upload.content_type or "").split(";", 1)[0].lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="The uploaded content type is not recognized as audio.")

    temp = tempfile.NamedTemporaryFile(prefix="voice-", suffix=suffix, delete=False)
    path = Path(temp.name)
    total = 0
    try:
        with temp:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"Audio must be {max_bytes // (1024 * 1024)} MB or smaller.")
                temp.write(chunk)
        if total == 0:
            raise HTTPException(status_code=422, detail="The uploaded audio file is empty.")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@router.post("", response_model=VoiceResponse, status_code=status.HTTP_200_OK)
async def voice(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None, max_length=128),
    language: str | None = Form(default=None, pattern="^(en|hi)$"),
    response_language: str | None = Form(default=None, pattern="^(en|hi)$"),
) -> VoiceResponse:
    settings = get_settings()
    if audio.content_type and audio.content_type.split(";", 1)[0].lower() not in ALLOWED_CONTENT_TYPES:
        await audio.close()
        raise HTTPException(status_code=415, detail="The uploaded content type is not recognized as audio.")
    temp_path = await _save_upload(audio, settings.audio_max_size_mb * 1024 * 1024)
    try:
        result = await request.app.state.pipeline.voice(
            temp_path, session_id, language, response_language=response_language
        )
    except SessionCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SpeechModelNotInstalledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceModelNotInstalledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OllamaUnavailableError, OllamaModelMissingError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TranscriptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OllamaResponseError, SpeechSynthesisError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected voice pipeline failure")
        raise HTTPException(status_code=500, detail="The local voice pipeline failed. Check the backend logs.") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return VoiceResponse(
        session_id=result.session_id,
        transcript=result.transcript,
        response=result.response,
        audio_url=f"/api/audio/{result.audio_path.name}" if result.audio_path else None,
        action=result.action,
        metrics=MetricsResponse(**result.metrics),
        created_at=result.created_at,
        language=result.language,
        language_confidence=result.language_confidence,
        speech_available=result.speech_available,
    )


@router.post("/stream")
async def voice_stream(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None, max_length=128),
    language: str | None = Form(default=None, pattern="^(en|hi)$"),
    response_language: str | None = Form(default=None, pattern="^(en|hi)$"),
) -> StreamingResponse:
    """NDJSON stream: transcript, text deltas, ordered chunked-Piper audio, then metrics."""
    settings = get_settings()
    temp_path = await _save_upload(audio, settings.audio_max_size_mb * 1024 * 1024)

    async def events():
        try:
            async for event in request.app.state.pipeline.voice_stream(
                temp_path, session_id, language, response_language=response_language
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except (SpeechModelNotInstalledError, VoiceModelNotInstalledError, OllamaUnavailableError,
                OllamaModelMissingError, TranscriptionError, OllamaResponseError, SpeechSynthesisError) as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"
        except Exception:
            logger.exception("Unexpected streaming voice pipeline failure")
            yield json.dumps({"type": "error", "detail": "The local streaming voice pipeline failed."}) + "\n"
        finally:
            temp_path.unlink(missing_ok=True)

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})
