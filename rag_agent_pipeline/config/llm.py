"""LLM configuration — Ollama via OpenAI-compatible API."""

import os

from langchain_openai import ChatOpenAI

import config.env  # noqa: F401  ensure .env is loaded

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "mistral:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "0")) or None  # 0 = provider default

LLM = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    streaming=True,
    openai_api_key="ollama",
    openai_api_base=OLLAMA_BASE_URL,
)
