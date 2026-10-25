"""Conservative text checks; these do not verify facts or Hindi grammar."""

import re
import unicodedata


HINDI_INSTRUCTION = (
    "Reply in accurate, natural Hindi using Devanagari. Understand Hindi written in Roman letters. "
    "When Hindi is selected, reply in Hindi and Devanagari even if the question is asked in English. "
    "Keep simple answers concise, but provide enough detail to fully answer questions that ask for an explanation. "
    "For common factual questions, give the answer rather than saying you are uncertain. If a transcript has an unfamiliar or malformed word, "
    "ask one short clarifying question instead of inventing an answer. Use history only for a relevant follow-up. "
    "Answer the latest question directly. Write city and person names in Devanagari, for example Chennai as चेन्नई."
)

# Only exact, recurring speech-recognition confusions belong here. This is not a
# general Hindi spellchecker: changing arbitrary words would be less reliable
# than asking the user to repeat themselves.
HINDI_STT_CORRECTIONS = {
    "\u092c\u093e\u0930\u0924": "\u092d\u093e\u0930\u0924",
    "\u092d\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940": "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940",
    "\u0930\u093e\u091c\u0926\u093e\u0928\u0940": "\u0930\u093e\u091c\u0927\u093e\u0928\u0940",
    "\u0930\u093e\u091c\u094d\u0926\u093e\u0928\u0940": "\u0930\u093e\u091c\u0927\u093e\u0928\u0940",
    "बारद": "भारत",
    "राज्टानी": "राजधानी",
}

LEGACY_HINDI_INSTRUCTION = (
    "1. If the user asks in Hindi — whether written in Devanagari (हिंदी) or in Roman/English letters "
    "(Hinglish, e.g. 'rajdhani kya hai?', 'rashtriya pashu') — reply in natural, grammatically correct "
    "Hindi using Devanagari script.\n"
    "2. In Hindi replies, write ALL proper nouns (cities, states, countries, names) in Devanagari "
    "(e.g. \u091a\u0947\u0928\u094d\u0928\u0908, \u0924\u092e\u093f\u0932\u0928\u093e\u0921\u0941, \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940, \u092d\u093e\u0930\u0924). Do not mix Latin letters into Hindi proper nouns.\n"
    "3. If the user asks in English, reply in simple, natural English.\n"
    "4. If the user explicitly requests a specific language, strictly follow that request.\n"
    "Roman Hindi Understanding:\n"
    "5. The user often writes Hindi phonetically in English letters with possible spelling variations "
    "(e.g. 'rajdhani', 'rajdani', 'rajtani'; 'rashtriya pashu', 'rastriya pashu'). "
    "Understand the intended Hindi meaning accurately from context.\n"
    "Accuracy:\n"
    "6. Answer the latest question directly with one concise spoken sentence for simple facts. "
    "Use prior turns only when relevant to a follow-up.\n"
    "7. Do not invent facts, definitions, or nonsensical words. "
    "If unsure, say: \u092e\u0941\u091d\u0947 \u0907\u0938 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u0928\u093f\u0936\u094d\u091a\u093f\u0924 \u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u0928\u0939\u0940\u0902 \u0939\u0948\u0964\n"
    "\u0938\u0930\u0932 \u0914\u0930 \u0938\u094d\u0935\u093e\u092d\u093e\u0935\u093f\u0915 \u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u0909\u0924\u094d\u0924\u0930 \u0926\u0947\u0902\u0964 \u0905\u0928\u0941\u092e\u093e\u0928 \u0932\u0917\u093e\u0915\u0930 \u0924\u0925\u094d\u092f \u0928 \u092c\u0928\u093e\u090f\u0902\u0964"
)


def normalize_text(text: str) -> str:
    """Keep letters, vowel marks, names and punctuation; never guess corrections."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_hindi_transcript(text: str) -> str:
    """Correct only known whole-word STT confusions before language-model use."""
    return re.sub(
        r"(?<!\S)(%s)(?!\S)" % "|".join(map(re.escape, HINDI_STT_CORRECTIONS)),
        lambda match: HINDI_STT_CORRECTIONS[match.group(1)],
        normalize_text(text),
    )


def text_quality_error(text: str) -> str | None:
    if not text.strip():
        return "empty text"
    if "\ufffd" in text or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        return "malformed text"
    if any(marker in text for marker in ("<|im_start|>", "<|im_end|>", "<tool_call>")):
        return "model control tokens"
    words = [word.casefold().strip(".,!?\u0964\u0965:;") for word in text.split()]
    # Four consecutive repeats of a word/short phrase are a clear decoding loop.
    for width in range(1, 5):
        for start in range(len(words) - width * 4 + 1):
            phrase = words[start:start + width]
            if any(phrase) and words[start:start + width * 4] == phrase * 4:
                return "repeated phrase loop"
    return None
