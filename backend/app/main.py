"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.voice import router as voice_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.llm import OllamaClient
from app.services.pipeline import VoicePipeline
from app.services.session_manager import SessionManager
from app.services.stt import SpeechToTextService
from app.services.tts import TextToSpeechService

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure lightweight services only; speech models are loaded on first use."""
    configure_logging()
    app.state.ollama = OllamaClient(settings)
    app.state.sessions = SessionManager(settings.session_max_turns, settings.session_max_count)
    app.state.stt = SpeechToTextService(settings)
    app.state.tts = TextToSpeechService(settings)
    app.state.pipeline = VoicePipeline(
        llm=app.state.ollama,
        sessions=app.state.sessions,
        stt=app.state.stt,
        tts=app.state.tts,
        voice_num_predict=settings.voice_num_predict,
        voice_hi_num_predict=settings.voice_hi_num_predict,
        hindi_num_predict=settings.ollama_hi_num_predict,
        hindi_model=settings.ollama_hi_model,
    )
    warmup = asyncio.create_task(app.state.ollama.warm_model())
    warmup.add_done_callback(
        lambda task: logger.warning("Ollama model warm-up failed: %s", task.exception())
        if not task.cancelled() and task.exception() else None
    )
    try:
        yield
    finally:
        if not warmup.done():
            warmup.cancel()
        await asyncio.gather(warmup, return_exceptions=True)
        await app.state.ollama.close()


app = FastAPI(
    title=settings.app_name,
    description="Local chat and speech pipeline backed by Ollama, Faster-Whisper, and Piper.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(voice_router, prefix="/api")

audio_dir = Path(settings.audio_output_dir)
app.mount("/api/audio", StaticFiles(directory=str(audio_dir), check_dir=False), name="generated-audio")


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"name": settings.app_name, "health": "/api/health", "docs": "/docs"}
