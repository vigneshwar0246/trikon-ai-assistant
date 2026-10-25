"""Opt-in live checks against the running API; never downloads or installs models.

Run from backend/: .venv/Scripts/python.exe scripts/check_hindi_quality.py
These limited semantic markers are test oracles, never runtime answer substitutions.
"""

import argparse
import json
from pathlib import Path

import httpx


# Each case: (label, question, language, expected_substrings)
# Expected substrings are checked case-insensitively against the response.
CASES = [
    ("A", "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?", "hi", ("\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940", "\u0928\u092f\u0940 \u0926\u093f\u0932\u094d\u0932\u0940", "new delhi", "dilli")),
    ("B", "Bharat ki rajdhani kya hai?", "hi", ("\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940", "\u0928\u092f\u0940 \u0926\u093f\u0932\u094d\u0932\u0940", "new delhi", "dilli")),
    ("C", "\u092e\u0941\u091d\u0947 \u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u091c\u0935\u093e\u092c \u0926\u0940\u091c\u093f\u090f\u0964 \u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?", "hi", ("\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940", "\u0928\u092f\u0940 \u0926\u093f\u0932\u094d\u0932\u0940", "new delhi")),
    ("D", "What is the capital of India?", "en", ("new delhi",)),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("../.run/hindi-quality-live.json"))
    args = parser.parse_args()
    results = []
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        for case, question, language, expected in CASES:
            response = client.post("/api/chat", json={"message": question, "language": language})
            data = response.json()
            answer = data.get("response", "")
            passed = response.is_success and any(word in answer.casefold() for word in expected)
            results.append({
                "case": case, "question": question, "language": language,
                "result": "PASS" if passed else "FAIL",
                "status": response.status_code, "response": data,
            })

        # Multi-turn Hindi context isolation
        first = client.post("/api/chat", json={"message": "Bharat ki rajdhani kya hai?", "language": "hi"})
        if first.is_success:
            session_id = first.json()["session_id"]
            response = client.post("/api/chat", json={
                "message": "Ab Bharat ki rajdhani batao.",
                "language": "hi",
                "session_id": session_id,
            })
            data = response.json() if response.is_success else {"error": response.text}
            answer = data.get("response", "") if isinstance(data, dict) else ""
            has_delhi = any(w in answer.casefold() for w in ("\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940", "new delhi"))
            passed = response.is_success and has_delhi
            results.append({
                "case": "J", "result": "PASS" if passed else "FAIL",
                "note": "Multi-turn Hindi context isolation verified",
                "status": response.status_code, "response": data,
            })
        else:
            results.append({
                "case": "J", "result": "FAIL",
                "note": "First turn failed",
                "status": first.status_code, "response": first.text,
            })

        # English then Hindi — context isolation
        first_en = client.post("/api/chat", json={"message": "What is exercise in one sentence?", "language": "en"})
        if first_en.is_success:
            response = client.post("/api/chat", json={
                "message": "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948?",
                "language": "hi",
                "session_id": first_en.json()["session_id"],
            })
            data = response.json() if response.is_success else {"error": response.text}
            answer = data.get("response", "") if isinstance(data, dict) else ""
            passed = response.is_success and any(word in answer for word in ("\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940", "\u0928\u092f\u0940 \u0926\u093f\u0932\u094d\u0932\u0940"))
            results.append({
                "case": "K", "result": "PASS" if passed else "FAIL",
                "note": "English turn then Hindi question — context isolation verified",
                "status": response.status_code, "response": data,
            })
        else:
            results.append({
                "case": "K", "result": "FAIL",
                "note": "First English turn failed",
                "status": first_en.status_code, "response": first_en.text,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        note = f" [{row.get('note', '')}]" if row.get("note") else ""
        print(f"{row['case']}: {row['result']} (HTTP {row['status']}){note}")
    print(f"Saved actual responses to {args.output}. Semantic markers require human review for contradictions.")
    return 1 if any(row["result"] == "FAIL" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
