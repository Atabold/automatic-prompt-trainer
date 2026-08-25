"""Scores a formatter output: language match, filler removal, and an LLM judge."""
import re
import threading

from langdetect import LangDetectException, detect

from app.core.config import load_config
from app.core.logging_setup import get_logger
from app.llm import client

logger = get_logger(__name__)

FILLER_PATTERN = re.compile(
    r"\b(um+|uh+|uhm+|euh+|ähm+|eh+|あの|えー|嗯|어|저)\b", re.IGNORECASE
)

_judge_cache: dict[int, float] = {}
_cache_lock = threading.Lock()
_cache_hits = 0

JUDGE_TEMPLATE = """You are an evaluator. Given a raw spoken/voice-input text, an expected formatted text, and an actual model output, rate the actual output from 0 to 100 based on the following criteria:

{criteria}

Raw input: {input_text}
Expected: {expected}
Actual output: {output}

Return only a number from 0 to 100."""


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def clear_judge_cache() -> None:
    global _cache_hits
    with _cache_lock:
        _judge_cache.clear()
        _cache_hits = 0


def judge_cache_stats() -> dict:
    with _cache_lock:
        return {"cached_evaluations": len(_judge_cache), "cache_hits": _cache_hits}


def llm_judge(input_text: str, expected: str, output: str) -> float:
    global _cache_hits
    cache_key = hash((input_text, expected, output))
    with _cache_lock:
        if cache_key in _judge_cache:
            _cache_hits += 1
            return _judge_cache[cache_key]

    criteria = load_config()["evaluation_criteria"]
    prompt = JUDGE_TEMPLATE.format(
        criteria=criteria, input_text=input_text, expected=expected, output=output
    )

    try:
        response, _ = client.chat(
            "judge",
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        content = (response.choices[0].message.content or "").strip()
        match = re.search(r"\d+(\.\d+)?", content)
        score = min(max(float(match.group()) / 100.0, 0.0), 1.0) if match else 0.0
    except client.ProviderNotConfigured:
        raise
    except Exception as exc:
        logger.warning("Judge call failed, scoring 0 for this case: %s", exc)
        score = 0.0

    with _cache_lock:
        _judge_cache[cache_key] = score
    return score


def score_case(case: dict, output: str) -> dict:
    """Return the composite score plus its components, so the UI can explain the number."""
    weights = load_config().get("weights", {"language": 0.3, "filler": 0.2, "judge": 0.5})

    detected = detect_language(output)
    language_score = 1.0 if detected == case.get("language") else 0.0
    filler_score = 0.0 if FILLER_PATTERN.search(output) else 1.0
    judge_score = llm_judge(case["input"], case["expected"], output)

    total = (
        weights.get("language", 0) * language_score
        + weights.get("filler", 0) * filler_score
        + weights.get("judge", 0) * judge_score
    )
    return {
        "score": total,
        "language_score": language_score,
        "filler_score": filler_score,
        "judge_score": judge_score,
        "detected_language": detected,
    }
