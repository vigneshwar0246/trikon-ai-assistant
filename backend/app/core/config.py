"""Application settings. No model or network work happens during import."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AI Voice Assistant API"
    environment: str = "development"
    log_level: str = "INFO"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_hi_model: str | None = None
    ollama_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    ollama_temperature: float = Field(default=0.1, ge=0, le=2)
    ollama_num_ctx: int = Field(default=1024, ge=512, le=32768)
    ollama_num_predict: int = Field(default=64, ge=16, le=1024)
    ollama_hi_num_predict: int = Field(default=256, ge=32, le=1024)
    voice_num_predict: int = Field(default=48, ge=16, le=256)
    voice_hi_num_predict: int = Field(default=192, ge=32, le=512)

    session_max_turns: int = Field(default=12, ge=1, le=100)
    session_max_count: int = Field(default=500, ge=1, le=10000)

    stt_model_path: Path = BACKEND_DIR / "models" / "faster-whisper-base"
    stt_multilingual_model_path: Path | None = None
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str | None = None
    stt_beam_size: int = Field(default=5, ge=1, le=10)
    stt_hi_beam_size: int = Field(default=5, ge=1, le=10)
    stt_language_confidence_threshold: float = Field(default=0.55, ge=0, le=1)
    audio_max_size_mb: int = Field(default=25, ge=1, le=100)
    audio_max_duration_seconds: int = Field(default=300, ge=1, le=1800)

    tts_model_path: Path = BACKEND_DIR / "models" / "piper" / "en_US-norman-medium.onnx"
    tts_config_path: Path | None = None
    tts_hi_model_path: Path | None = None
    tts_hi_config_path: Path | None = None
    audio_output_dir: Path = BACKEND_DIR / "generated_audio"
    tts_use_cuda: bool = False

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator(
        "stt_model_path", "stt_multilingual_model_path", "tts_model_path", "audio_output_dir", "tts_config_path",
        "tts_hi_model_path", "tts_hi_config_path",
        mode="before",
    )
    @classmethod
    def resolve_local_paths(cls, value: str | Path | None) -> Path | None:
        if value is None or value == "":
            return None
        path = Path(value)
        return path if path.is_absolute() else BACKEND_DIR / path

    @field_validator("ollama_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
