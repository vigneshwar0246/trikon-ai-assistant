from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.config import Settings
from app.services.pipeline import VoicePipeline
from app.services.session_manager import SessionManager
from app.services.stt import TranscriptionError
from app.services.stt import SpeechToTextService
from app.services.tts import TextToSpeechService, VoiceModelNotInstalledError


def test_language_voice_mapping_never_falls_back_to_english(tmp_path: Path) -> None:
    english = tmp_path / "english.onnx"
    english.write_bytes(b"model")
    Path(f"{english}.json").write_text("{}", encoding="utf-8")
    service = TextToSpeechService(Settings(tts_model_path=english, tts_hi_model_path=None))

    assert service.available_for("en") is True
    assert service.available_for("hi") is False
    with pytest.raises(VoiceModelNotInstalledError, match="'hi'"):
        service._get_voice("hi")


def test_sentence_chunking_preserves_safe_boundaries() -> None:
    chunk, remainder = VoicePipeline._take_speakable_chunk("This is the first sentence. This remains")
    assert chunk == "This is the first sentence."
    assert remainder == "This remains"
    assert VoicePipeline._take_speakable_chunk("an unfinished short phrase") == (None, "an unfinished short phrase")
    assert VoicePipeline._take_speakable_chunk("अंतिम वाक्य। अगला भाग")[0] == "अंतिम वाक्य।"


def test_manual_hindi_sets_decoder_language_without_prompt_leakage(tmp_path: Path) -> None:
    model = Mock()
    model.transcribe.return_value = (
        [Mock(text=" नमस्ते ")],
        Mock(language="hi", language_probability=0.99, duration=1.0),
    )
    service = SpeechToTextService(Settings())
    service.settings.stt_multilingual_model_path = None
    service._model = model

    text, _ = service.transcribe(tmp_path / "hindi.wav", "hi")

    assert text == "नमस्ते"
    assert model.transcribe.call_args.kwargs["initial_prompt"] is None
    assert model.transcribe.call_args.kwargs["language"] == "hi"
    assert model.transcribe.call_args.kwargs["condition_on_previous_text"] is False


def test_hindi_stream_starts_earlier_without_splitting_words() -> None:
    text = "पानी बचाने के लिए नल बंद रखें और बारिश का पानी इकट्ठा करके पौधों में इस्तेमाल करें"
    chunk, remainder = VoicePipeline._take_speakable_chunk(text, language="hi")
    assert chunk is not None
    assert f"{chunk} {remainder}" == text
    assert VoicePipeline._take_speakable_chunk(text, language="en") == (None, text)
    assert VoicePipeline._take_speakable_chunk("क" * 81, language="hi") == (None, "क" * 81)


@pytest.mark.asyncio
async def test_uncertain_automatic_language_requires_manual_selection(tmp_path: Path) -> None:
    stt = Mock(
        transcribe=Mock(return_value=("short", {"language": "hi", "language_probability": 0.2})),
        settings=Settings(stt_language_confidence_threshold=0.55),
    )
    tts = Mock(available_for=Mock(return_value=True))
    pipeline = VoicePipeline(llm=Mock(chat=AsyncMock()), sessions=SessionManager(), stt=stt, tts=tts)

    with pytest.raises(TranscriptionError, match="uncertain"):
        await pipeline.voice(tmp_path / "short.wav")


@pytest.mark.asyncio
async def test_manual_language_persists_in_session_prompt() -> None:
    llm = Mock(chat=AsyncMock(return_value=("नमस्ते", {"total_duration": 1_000_000})))
    sessions = SessionManager()
    pipeline = VoicePipeline(llm=llm, sessions=sessions)

    await pipeline.chat("hello", "language-session", "hi")
    await pipeline.chat("again", "language-session")

    second_messages = llm.chat.await_args_list[1].args[0]
    assert second_messages[0]["role"] == "system"
    assert "Hindi" in second_messages[0]["content"]


@pytest.mark.asyncio
async def test_english_voice_question_can_request_hindi_response(tmp_path: Path) -> None:
    """Response language must not force Faster-Whisper to decode English as Hindi."""
    stt = Mock(
        transcribe=Mock(return_value=(
            "Tell me the capital of India in Hindi",
            {"language": "en", "language_probability": 0.99},
        )),
        settings=Settings(),
    )
    llm = Mock(chat=AsyncMock(return_value=("भारत की राजधानी नई दिल्ली है।", {"total_duration": 1_000_000})))
    tts = Mock(available_for=Mock(return_value=True), synthesize_to_file=Mock(return_value=(tmp_path / "reply.wav", 1.0)))
    pipeline = VoicePipeline(llm=llm, sessions=SessionManager(), stt=stt, tts=tts)

    result = await pipeline.voice(tmp_path / "question.wav", response_language="hi")

    assert stt.transcribe.call_args.args[1] is None
    assert result.language == "hi"
    assert "नई दिल्ली" in result.response
    assert "even if the question is asked in English" in llm.chat.await_args.args[0][0]["content"]
    assert tts.synthesize_to_file.call_args.args[1] == "hi"


@pytest.mark.asyncio
async def test_cancelled_stream_does_not_commit_partial_assistant_turn(tmp_path: Path) -> None:
    class StreamingLlm:
        async def chat_stream(self, messages, *, num_predict):
            yield {"token": "A partial response without a boundary"}
            yield {"done": True, "metadata": {}}

    stt = Mock(
        transcribe=Mock(return_value=("hello", {"language": "en", "language_probability": 0.99})),
        settings=Settings(),
    )
    tts = Mock(available_for=Mock(return_value=True))
    sessions = SessionManager()
    pipeline = VoicePipeline(llm=StreamingLlm(), sessions=sessions, stt=stt, tts=tts)
    stream = pipeline.voice_stream(tmp_path / "input.wav", "cancelled", "en")

    assert (await stream.__anext__())["type"] == "transcript"
    assert (await stream.__anext__())["type"] == "text"
    await stream.aclose()

    summaries = await sessions.list_sessions()
    assert summaries[0][0] == "cancelled"
    assert summaries[0][1] == 0
