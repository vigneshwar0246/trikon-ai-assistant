"""Lazy local Piper speech synthesis. Voice files are never downloaded here."""

import logging
import re
import threading
import time
import unicodedata
import wave
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class VoiceModelNotInstalledError(RuntimeError):
    pass


class SpeechSynthesisError(RuntimeError):
    pass


def to_spoken_text(text: str) -> str:
    """Remove presentation markup and code that should not be read aloud."""
    includes_code = bool(re.search(r"```|`[^`]+`", text))
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\b(?:https?://|www\.)\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", text)
    text = re.sub(r"[*_~#|]+", "", text)
    text = text.replace("&", " and ").replace("+", " plus ").replace("=", " equals ")
    text = text.replace("@", " at ").replace("%", " percent ").replace("/", " or ")
    text = "".join(character for character in text if not unicodedata.category(character).startswith("S"))
    text = " ".join(text.split())
    if includes_code:
        text = f"{text} I've included the code in the text response.".strip()
    return text or "I do not have a spoken response for that yet."


class TextToSpeechService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._voices: dict[str, object] = {}
        self._voice = None  # Backward-compatible English injection used by focused tests.
        self._voice_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.available_for("en")

    def voice_paths(self, language: str) -> tuple[Path | None, Path | None]:
        if language == "hi":
            model_path, config_path = self.settings.tts_hi_model_path, self.settings.tts_hi_config_path
        else:
            model_path, config_path = self.settings.tts_model_path, self.settings.tts_config_path
        if model_path is not None and config_path is None:
            config_path = Path(f"{model_path}.json")
        return model_path, config_path

    def available_for(self, language: str) -> bool:
        model_path, config_path = self.voice_paths(language)
        if model_path is None or config_path is None:
            return False
        return model_path.is_file() and config_path.is_file() and find_spec("piper") is not None

    def language_availability(self) -> dict[str, bool]:
        return {language: self.available_for(language) for language in ("en", "hi")}

    def _get_voice(self, language: str = "en"):
        if language == "en" and self._voice is not None:
            return self._voice
        if language in self._voices:
            return self._voices[language]
        with self._voice_lock:
            if language not in self._voices:
                model_path, config_path = self.voice_paths(language)
                if model_path is None or config_path is None or not model_path.is_file() or not config_path.is_file():
                    raise VoiceModelNotInstalledError(
                        f"A compatible local Piper voice for '{language}' is not configured. "
                        "Set its TTS language model and config paths; no English or cloud fallback was used."
                    )
                try:
                    from piper import PiperVoice
                except ImportError as exc:
                    raise VoiceModelNotInstalledError("piper-tts is not installed in this environment.") from exc
                try:
                    self._voices[language] = PiperVoice.load(
                        model_path,
                        config_path=config_path,
                        use_cuda=self.settings.tts_use_cuda,
                        download_dir=self.settings.audio_output_dir,
                    )
                except Exception as exc:
                    logger.exception("Could not load the configured Piper voice")
                    raise VoiceModelNotInstalledError("The Piper voice could not be loaded. Check its model and config files.") from exc
        return self._voices[language]

    def prepare(self, language: str) -> None:
        """Load the selected speech voice before a reply needs to be played."""
        self._get_voice(language)

    def synthesize_to_file(self, text: str, language: str = "en") -> tuple[Path, float]:
        """Create one WAV file and return its path and measured synthesis time."""
        from time import perf_counter

        self.settings.audio_output_dir.mkdir(parents=True, exist_ok=True)
        voice = self._get_voice(language)
        self._remove_expired_audio()
        output_path = self.settings.audio_output_dir / f"{uuid4().hex}.wav"
        started = perf_counter()
        try:
            from piper.config import SynthesisConfig

            with wave.open(str(output_path), "wb") as wav_file:
                voice.synthesize_wav(to_spoken_text(text), wav_file, syn_config=SynthesisConfig(length_scale=0.92))
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            logger.exception("Piper speech synthesis failed")
            raise SpeechSynthesisError("The local TTS engine could not synthesize this response.") from exc
        return output_path, round((perf_counter() - started) * 1000, 2)

    def _remove_expired_audio(self) -> None:
        """Prune expired and excess generated WAVs to keep the local cache bounded."""
        cutoff = time.time() - 3600
        files = []
        for path in self.settings.audio_output_dir.glob("*.wav"):
            try:
                if path.is_file():
                    files.append((path.stat().st_mtime, path))
            except OSError:
                logger.warning("Could not inspect generated audio file")
        for index, (modified_at, path) in enumerate(sorted(files, key=lambda item: item[0], reverse=True)):
            try:
                if index >= 100 or modified_at < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove expired generated audio file")


tts_service = TextToSpeechService()
