from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.config import Settings
from app.services.language_quality import normalize_hindi_transcript, normalize_text, text_quality_error, HINDI_INSTRUCTION
from app.services.llm import OllamaResponseError, OllamaUnavailableError
from app.services.pipeline import VoicePipeline
from app.services.session_manager import SessionManager
from app.services.stt import SpeechToTextService, TranscriptionError
from app.services.tts import to_spoken_text


# ---------------------------------------------------------------------------
# Pipeline contract test cases (mocked LLM; answers are provided by the test)
# These verify routing, prompt construction, TTS invocation, and token limits.
# They are NOT claims that a real model produces these exact answers.
# ---------------------------------------------------------------------------

CASES = [
    # (question, language, mocked_answer, expected_meaning)
    # 1. Devanagari: Bharat ki rajdhani — New Delhi
    ("\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?", "hi", "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964", "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940"),
    # 2. Devanagari: Bharat ki rajdhani ka naam
    ("\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u093e \u0928\u093e\u092e \u0915\u094d\u092f\u093e \u0939\u0948?", "hi", "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0939\u0948\u0964", "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940"),
    # 3. Roman Hindi: Bharat ki rajdhani kya hai
    ("Bharat ki rajdhani kya hai?", "hi", "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964", "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940"),
    # 4. Explicit Hindi request
    ("\u092e\u0941\u091d\u0947 \u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u091c\u0935\u093e\u092c \u0926\u0940\u091c\u093f\u090f\u0964 \u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?", "hi", "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964", "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940"),
    # 5. English question about India capital
    ("What is the capital of India?", "en", "The capital of India is New Delhi.", "New Delhi"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("question,language,answer,meaning", CASES)
async def test_factual_answer_passes_unchanged_to_display_and_correct_voice(question, language, answer, meaning):
    # Contract test: verifies routing, prompts, TTS call, and token limits.
    # NOT evidence that a real model knows these facts.
    llm = Mock(chat=AsyncMock(return_value=(answer, {"done_reason": "stop"})))
    stt = Mock(transcribe=Mock(return_value=(question, {"language": language, "language_probability": 1})))
    tts = Mock(synthesize_to_file=Mock(return_value=(Path("reply.wav"), 1)), available_for=Mock(return_value=True))
    pipeline = VoicePipeline(llm=llm, stt=stt, tts=tts, sessions=SessionManager())
    result = await pipeline.voice("input.wav", language=language)
    assert meaning.casefold() in result.response.casefold()
    assert result.transcript == question
    assert llm.chat.await_args.args[0][-1]["content"] == question
    assert tts.synthesize_to_file.call_args.args[0] == result.response
    if language == "hi":
        assert tts.synthesize_to_file.call_args.args[1] == "hi"
        assert "Devanagari" in llm.chat.await_args.args[0][0]["content"]
        assert llm.chat.await_args.kwargs["num_predict"] == 96
    else:
        assert llm.chat.await_args.kwargs["num_predict"] == 80


@pytest.mark.asyncio
async def test_roman_hindi_question_receives_hindi_system_prompt():
    """Roman Hindi must use the Hindi system prompt with Devanagari rules."""
    answer = "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964"
    llm = Mock(chat=AsyncMock(return_value=(answer, {"done_reason": "stop"})))
    stt = Mock(transcribe=Mock(return_value=("Bharat ki rajdhani kya hai?", {"language": "hi", "language_probability": 0.95})))
    tts = Mock(synthesize_to_file=Mock(return_value=(Path("reply.wav"), 1)), available_for=Mock(return_value=True))
    pipeline = VoicePipeline(llm=llm, stt=stt, tts=tts, sessions=SessionManager())
    result = await pipeline.voice("input.wav", language="hi")
    system_content = llm.chat.await_args.args[0][0]["content"]
    # System prompt must include Roman Hindi understanding instruction
    assert "Roman" in system_content or "Hinglish" in system_content or "roman" in system_content.lower()
    assert "Devanagari" in system_content
    assert "\u091a\u0947\u0928\u094d\u0928\u0908" in result.response  # चेन्नई


@pytest.mark.asyncio
async def test_hindi_instruction_contains_roman_hindi_understanding():
    """HINDI_INSTRUCTION must explicitly mention Roman Hindi/Hinglish understanding."""
    assert "Roman" in HINDI_INSTRUCTION or "Hinglish" in HINDI_INSTRUCTION
    assert "Devanagari" in HINDI_INSTRUCTION
    # Must instruct proper noun Devanagari
    assert "\u091a\u0947\u0928\u094d\u0928\u0908" in HINDI_INSTRUCTION or "proper noun" in HINDI_INSTRUCTION.lower()


@pytest.mark.asyncio
async def test_language_switch_preserves_context_but_prioritizes_latest_question():
    llm = Mock(chat=AsyncMock(side_effect=[("Exercise can improve fitness.", {}), ("\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964", {})]))
    pipeline = VoicePipeline(llm=llm, sessions=SessionManager())
    await pipeline.chat("Tell me about exercise", "mixed", "en")
    result = await pipeline.chat(CASES[0][0], "mixed", "hi")
    messages = llm.chat.await_args.args[0]
    assert messages[-1]["content"] == CASES[0][0]
    assert "latest question" in messages[0]["content"] or "latest" in messages[0]["content"].lower()
    assert "Devanagari" in messages[0]["content"]
    assert any(m["content"] == "Exercise can improve fitness." for m in messages)
    assert "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940" in result.response


@pytest.mark.asyncio
async def test_typed_hindi_without_override_is_not_locked_to_previous_english():
    llm = Mock(chat=AsyncMock(return_value=("\u0909\u0924\u094d\u0924\u0930", {})))
    pipeline = VoicePipeline(llm=llm, sessions=SessionManager())
    await pipeline.chat("Hello", "auto", "en")
    await pipeline.chat(CASES[0][0], "auto")
    assert "Devanagari" in llm.chat.await_args.args[0][0]["content"]


@pytest.mark.parametrize("text", ["", "\u092c\u093e\u0930 \u092c\u093e\u0930 \u092c\u093e\u0930 \u092c\u093e\u0930", "\u092f\u0939 \u0938\u0939\u0940 \u0939\u0948 \u092f\u0939 \u0938\u0939\u0940 \u0939\u0948 \u092f\u0939 \u0938\u0939\u0940 \u0939\u0948 \u092f\u0939 \u0938\u0939\u0940 \u0939\u0948", "\u0909\u0924\u094d\u0924\u0930\ufffd", "<|im_start|>assistant"])
def test_obvious_bad_text_is_rejected(text):
    assert text_quality_error(text)


@pytest.mark.parametrize("text", ["\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940\u0964", "Python 3.11 \u092e\u0947\u0902 2 files \u0939\u0948\u0902\u0964", "NASA \u0915\u093e \u092e\u093f\u0936\u0928 2026 \u092e\u0947\u0902 \u0939\u0948\u0964", "New Delhi", "42"])
def test_names_numbers_and_mixed_language_are_allowed(text):
    assert text_quality_error(text) is None
    assert to_spoken_text(text) == text


def test_normalization_preserves_devanagari_marks_and_punctuation():
    assert normalize_text("  \u092d\u093e\u0930\u0924 \u0915\u0940\n\u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964  ") == "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,metadata", [("\u092c\u093e\u0930 " * 8, {}), ("\u0905\u0927\u0942\u0930\u093e \u0909\u0924\u094d\u0924\u0930", {"done_reason": "length"}), ("", {})])
async def test_invalid_hindi_stream_never_reaches_tts_or_history(answer, metadata):
    class Llm:
        async def chat_stream(self, messages, *, num_predict):
            yield {"token": answer}
            yield {"done": True, "metadata": metadata}
    stt = Mock(transcribe=Mock(return_value=(CASES[0][0], {"language": "hi"})))
    tts = Mock(available_for=Mock(return_value=True))
    sessions = SessionManager()
    pipeline = VoicePipeline(llm=Llm(), stt=stt, tts=tts, sessions=sessions)
    events = []
    with pytest.raises(OllamaResponseError):
        async for event in pipeline.voice_stream("input.wav", "invalid", "hi"):
            events.append(event)
    assert [event["type"] for event in events] == ["transcript"]
    tts.synthesize_to_file.assert_not_called()
    assert (await sessions.list_sessions())[0][1] == 0


@pytest.mark.asyncio
async def test_valid_hindi_stream_uses_same_answer_for_text_and_speech():
    answer = "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964"
    class Llm:
        async def chat_stream(self, messages, *, num_predict):
            yield {"token": "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 "}
            yield {"token": "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 \u0939\u0948\u0964", "done": True, "metadata": {"done_reason": "stop"}}
    stt = Mock(transcribe=Mock(return_value=(CASES[0][0], {"language": "hi"})))
    tts = Mock(available_for=Mock(return_value=True), synthesize_to_file=Mock(return_value=(Path("reply.wav"), 1)))
    pipeline = VoicePipeline(llm=Llm(), stt=stt, tts=tts, sessions=SessionManager())
    events = [event async for event in pipeline.voice_stream("input.wav", language="hi")]
    assert events[-1]["response"] == answer
    assert "".join(e["delta"] for e in events if e["type"] == "text") == answer
    tts.synthesize_to_file.assert_called_once_with(answer, "hi")


@pytest.mark.asyncio
async def test_model_timeout_does_not_speak_or_commit_history():
    llm = Mock(chat=AsyncMock(side_effect=OllamaUnavailableError("Timed out; please retry.")))
    tts = Mock()
    stt = Mock(transcribe=Mock(return_value=(CASES[0][0], {"language": "hi"})))
    pipeline = VoicePipeline(llm=llm, stt=stt, tts=tts, sessions=SessionManager())
    with pytest.raises(OllamaUnavailableError):
        await pipeline.voice("input.wav", "timeout", "hi")
    tts.synthesize_to_file.assert_not_called()
    assert (await pipeline.sessions.list_sessions())[0][1] == 0


@pytest.mark.asyncio
async def test_uncertain_transcript_never_reaches_model_or_tts():
    stt = Mock(transcribe=Mock(side_effect=TranscriptionError("Please repeat clearly.")))
    llm, tts = Mock(), Mock()
    pipeline = VoicePipeline(llm=llm, stt=stt, tts=tts, sessions=SessionManager())
    with pytest.raises(TranscriptionError, match="repeat"):
        await pipeline.voice("input.wav", "uncertain", "hi")
    llm.chat.assert_not_called()
    tts.synthesize_to_file.assert_not_called()
    assert (await pipeline.sessions.list_sessions())[0][1] == 0


@pytest.mark.parametrize("logprob,compression", [(-1.5, 1.1), (-0.2, 3.0)])
def test_low_quality_stt_rejected_even_with_explicit_hindi(logprob, compression):
    service = SpeechToTextService(Settings())
    model = Mock()
    model.transcribe.return_value = ([SimpleNamespace(text="\u0938\u0902\u0926\u093f\u0917\u094d\u0927", avg_logprob=logprob, compression_ratio=compression)], SimpleNamespace(duration=1))
    with pytest.raises(TranscriptionError, match="uncertain"):
        service._transcribe_with_model(model, "audio.wav", "hi")


@pytest.mark.parametrize("language,probability,expected_calls", [("hi", 0.95, 2), ("hi", 0.2, 1), ("en", 0.99, 1)])
def test_auto_detection_only_routes_confident_hindi_to_hindi_model(language, probability, expected_calls):
    service = SpeechToTextService(Settings(stt_multilingual_model_path=Path("small")))
    service._get_model_for_language = Mock(return_value=object())
    service._transcribe_with_model = Mock(return_value=("speech", SimpleNamespace(language=language, language_probability=probability)))
    service.transcribe("input.wav")
    assert service._get_model_for_language.call_count == expected_calls
    assert service._get_model_for_language.call_args.args[0] == ("hi" if expected_calls == 2 else None)


def test_known_hindi_stt_confusions_are_corrected_as_whole_words_only():
    noisy = "\u092c\u093e\u0930\u0926 \u0915\u0940 \u0930\u093e\u091c\u094d\u091f\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?"
    corrected = "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?"
    assert normalize_hindi_transcript(noisy) == corrected
    assert normalize_hindi_transcript("\u092c\u093e\u0930\u0926\u093f\u0936 \u0930\u093e\u091c\u094d\u091f\u093e\u0928\u0940") == "\u092c\u093e\u0930\u0926\u093f\u0936 \u0930\u093e\u091c\u0927\u093e\u0928\u0940"
