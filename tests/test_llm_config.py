import os
from unittest.mock import patch
from llm_config import get_groq_model, has_groq_api_key

def test_get_groq_model_default():
    with patch.dict(os.environ, {}, clear=True):
        assert get_groq_model() == "llama-3.1-8b-instant"

def test_get_groq_model_custom():
    with patch.dict(os.environ, {"GROQ_MODEL": "llama3-70b-8192"}):
        assert get_groq_model() == "llama3-70b-8192"

def test_has_groq_api_key_true():
    with patch.dict(os.environ, {"GROQ_API_KEY": "some-key"}):
        assert has_groq_api_key() is True

def test_has_groq_api_key_false():
    with patch.dict(os.environ, {}, clear=True):
        assert has_groq_api_key() is False

def test_has_groq_api_key_empty():
    with patch.dict(os.environ, {"GROQ_API_KEY": "  "}):
        assert has_groq_api_key() is False
