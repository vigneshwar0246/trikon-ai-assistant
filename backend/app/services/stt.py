"""Lazy Faster-Whisper speech recognition with local-only model loading."""

import logging
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from app.core.cuda import configure_cuda_dll_directories
from app.core.config import Settings, get_settings
from app.services.language_quality import normalize_hindi_transcript, normalize_text, text_quality_error

logger = logging.getLogger(__name__)


class SpeechModelNotInstalledError(RuntimeError):
    pass


class TranscriptionError(RuntimeError):
    pass


class SpeechToTextService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any = None
        self._language_models: dict[str, Any] = {}
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.active_device = self.settings.stt_device
        self.active_compute_type = self.settings.stt_compute_type
        self.fallback_reason: str | None = None
        self.last_successful_device: str | None = None

    @property
    def available(self) -> bool:
        model_path = self.settings.stt_model_path
        return (
            model_path.is_dir()
            and (model_path / "model.bin").is_file()
            and (model_path / "config.json").is_file()
            and ((model_path / "vocabulary.txt").is_file() or (model_path / "vocabulary.json").is_file())
            and find_spec("faster_whisper") is not None
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                model_path = self.settings.stt_model_path
                if not self.available:
                    raise SpeechModelNotInstalledError(
                        f"Faster-Whisper package or required model files (model.bin, config.json, vocabulary) are missing at '{model_path}'. Follow docs/architecture.md."
                    )
                if self.settings.stt_device.lower() == "cuda":
                    try:
                        self._model = self._load_model("cuda", self.settings.stt_compute_type)
                    except Exception as exc:
                        self._fallback_to_cpu(exc)
                else:
                    self._model = self._load_model(self.settings.stt_device, self.settings.stt_compute_type)
        return self._model

    def _get_model_for_language(self, language: str | None) -> Any:
        path = self.settings.stt_multilingual_model_path
        if language != "hi" or path is None:
            return self._get_model()
        if language in self._language_models:
            return self._language_models[language]
        with self._model_lock:
            if language not in self._language_models:
                required = (path / "model.bin", path / "config.json")
                if not all(item.is_file() for item in required):
                    raise SpeechModelNotInstalledError(
                        f"The configured multilingual Faster-Whisper model is incomplete at '{path}'."
                    )
                try:
                    if self.active_device == "cuda":
                        configure_cuda_dll_directories()
                    from faster_whisper import WhisperModel
                    self._language_models[language] = WhisperModel(
                        str(path), device=self.active_device, compute_type=self.active_compute_type
                    )
                except Exception as exc:
                    if self.active_device != "cuda":
                        raise
                    self._fallback_to_cpu(exc)
                    self._language_models[language] = WhisperModel(str(path), device="cpu", compute_type="int8")
        return self._language_models[language]

    def _load_model(self, device: str, compute_type: str) -> Any:
        if device.lower() == "cuda":
            configure_cuda_dll_directories()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SpeechModelNotInstalledError("faster-whisper is not installed in this environment.") from exc
        # A directory is passed intentionally: CTranslate2 loads local files and does not contact Hugging Face.
        model = WhisperModel(
            str(self.settings.stt_model_path),
            device=device,
            compute_type=compute_type,
        )
        self.active_device = device
        self.active_compute_type = compute_type
        return model

    def _fallback_to_cpu(self, reason: Exception) -> None:
        self.fallback_reason = f"{type(reason).__name__}: {reason}"[:300]
        logger.warning("CUDA STT failed; retrying with CPU/int8: %s", self.fallback_reason)
        self._model = self._load_model("cpu", "int8")
        self._language_models.clear()

    @staticmethod
    def _is_cuda_runtime_error(error: Exception) -> bool:
        detail = str(error).lower()
        return any(token in detail for token in (
            "cuda", "cublas", "cudnn", "gpu driver", "no kernel image",
        ))

    def _transcribe_with_model(self, model: Any, audio_path: str | Path, language: str | None) -> tuple[str, Any]:
        selected_language = language or self.settings.stt_language
        segments, info = model.transcribe(
            str(audio_path),
            language=selected_language,
            beam_size=self.settings.stt_hi_beam_size if selected_language == "hi" else self.settings.stt_beam_size,
            vad_filter=True,
            # Language is a decoder option; a prose prompt can leak into uncertain speech.
            initial_prompt=None,
            condition_on_previous_text=False,
        )
        duration = getattr(info, "duration", None)
        if duration is not None and duration > self.settings.audio_max_duration_seconds:
            raise TranscriptionError(
                f"Audio is longer than the {self.settings.audio_max_duration_seconds}-second limit."
            )
        # Segment iteration performs inference, so keep it inside this service call.
        segments = list(segments)
        for segment in segments:
            logprob = getattr(segment, "avg_logprob", None)
            compression = getattr(segment, "compression_ratio", None)
            if ((isinstance(logprob, (int, float)) and logprob < -1.0)
                    or (isinstance(compression, (int, float)) and compression > 2.4)):
                raise TranscriptionError(
                    "Speech recognition was uncertain. Please repeat clearly or type your question."
                )
        text = normalize_text(" ".join(segment.text.strip() for segment in segments))
        if not text:
            raise TranscriptionError("No speech was detected in the uploaded audio.")
        if text_quality_error(text):
            raise TranscriptionError("Speech recognition was unclear. Please repeat or type your question.")
        if selected_language == "hi":
            text = normalize_hindi_transcript(text)
        return text, info

    def _decode(self, audio_path: str | Path, language: str | None) -> tuple[str, Any]:
        model = self._get_model_for_language(language)
        try:
            return self._transcribe_with_model(model, audio_path, language)
        except Exception as exc:
            if self.active_device != "cuda" or not self._is_cuda_runtime_error(exc):
                raise
            self._fallback_to_cpu(exc)
            model = self._get_model_for_language(language)
            return self._transcribe_with_model(model, audio_path, language)

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> tuple[str, dict[str, Any]]:
        try:
            with self._inference_lock:
                text, info = self._decode(audio_path, language or self.settings.stt_language)
                detected_probability = getattr(info, "language_probability", None)
                # Auto detection starts with the base model, then uses the Hindi model
                # only after a confident Hindi detection. English stays on the base model.
                if (not (language or self.settings.stt_language)
                        and getattr(info, "language", None) == "hi"
                        and isinstance(getattr(info, "language_probability", None), (int, float))
                        and info.language_probability >= self.settings.stt_language_confidence_threshold
                        and self.settings.stt_multilingual_model_path is not None):
                    text, info = self._decode(audio_path, "hi")
        except TranscriptionError:
            raise
        except SpeechModelNotInstalledError:
            raise
        except Exception as exc:  # decoder errors vary by uploaded container/codec
            logger.exception("Speech transcription failed")
            raise TranscriptionError("The audio could not be transcribed. Use a supported audio file and try again.") from exc
        self.last_successful_device = self.active_device
        return text, {
            "language": getattr(info, "language", None),
            "language_probability": detected_probability,
        }


stt_service = SpeechToTextService()
