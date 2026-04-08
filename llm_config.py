from __future__ import annotations

import os


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq_model() -> str:
    value = os.getenv("GROQ_MODEL")
    if value is None:
        return DEFAULT_GROQ_MODEL

    value = value.strip()
    return value or DEFAULT_GROQ_MODEL


def has_groq_api_key() -> bool:
    value = os.getenv("GROQ_API_KEY")
    return bool(value and value.strip())
