"""Transparent, conservative normalization for common chat abbreviations."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = ROOT / "data" / "slang_lexicon.json"


@lru_cache(maxsize=1)
def slang_lexicon() -> dict[str, str]:
    try:
        return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def normalize_slang(text: str) -> str:
    """Expand exact chat shorthand and reduce elongated characters conservatively.

    Terms are whole-token matches only. Keep the original text separately for
    review; slang mappings are suggestions and can miss local meaning.
    """
    value = unicodedata.normalize("NFKC", str(text or "")).replace("’", "'")
    for shorthand, expansion in sorted(slang_lexicon().items(), key=lambda pair: -len(pair[0])):
        value = re.sub(rf"(?<!\w){re.escape(shorthand)}(?!\w)", expansion, value, flags=re.IGNORECASE)
    # Keep two repeated letters to preserve some emphasis (e.g., "sooo" -> "soo").
    value = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", value)
    return value


def normalize_documents(texts):
    return [normalize_slang(text) for text in texts]


_HINGLISH = {"hai", "hain", "nahi", "nahin", "nhi", "kya", "kyun", "kyu", "kaise", "mera", "meri", "tum", "aap", "log", "mat", "yaar", "accha", "acha", "bahut"}
_BANGLISH = {"ami", "tumi", "apni", "bhalo", "kemon", "keno", "na", "onek", "amar", "tomar", "ki", "hobe", "ache"}
_MARATHI = {"आहे", "आणि", "मला", "माझा", "माझी", "तुम्ही", "नाही", "काय"}


def detect_language(text: str) -> str:
    """A lightweight script/keyword hint, not a formal language identifier."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    devanagari = len(re.findall(r"[\u0900-\u097f]", value))
    bengali = len(re.findall(r"[\u0980-\u09ff]", value))
    arabic = len(re.findall(r"[\u0600-\u06ff]", value))
    han = len(re.findall(r"[\u4e00-\u9fff]", value))
    latin_words = set(re.findall(r"[a-z]+", value.lower()))
    indic_words = set(re.findall(r"[\w]+", value.lower()))
    if bengali and re.search(r"[A-Za-z]", value):
        return "Bengali-English code-mixed"
    if bengali:
        return "Bengali"
    if devanagari:
        script_hint = "Marathi" if indic_words.intersection(_MARATHI) else "Hindi"
        return f"{script_hint}-English code-mixed" if re.search(r"[A-Za-z]", value) else f"{script_hint} / Devanagari"
    if arabic:
        return "Arabic / Urdu script"
    if han:
        return "Chinese"
    if re.search(r"[\u0b80-\u0bff]", value):
        return "Tamil script"
    if re.search(r"[\u0c00-\u0c7f]", value):
        return "Telugu script"
    if re.search(r"[\u0a80-\u0aff]", value):
        return "Gujarati script"
    if re.search(r"[\u0a00-\u0a7f]", value):
        return "Punjabi / Gurmukhi script"
    if re.search(r"[\u0c80-\u0cff]", value):
        return "Kannada script"
    if re.search(r"[\u0d00-\u0d7f]", value):
        return "Malayalam script"
    if len(latin_words.intersection(_HINGLISH)) >= 2:
        return "Hindi-English code-mixed"
    if len(latin_words.intersection(_BANGLISH)) >= 2:
        return "Bengali-English code-mixed (Romanized hint)"
    if latin_words:
        return "English / Latin-script"
    return "Unknown / mixed"
