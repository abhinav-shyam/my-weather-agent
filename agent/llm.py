from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent.schema import WeatherIntent

load_dotenv()

# LLM config constants (retries, concurrency, etc.)

LLM_MAX_RETRIES = 5
LLM_RETRY_BACKOFF_SECONDS = 1.0
LLM_CONCURRENCY_LIMIT = 3
LLM_REQUEST_TIMEOUT_SECONDS = 30
_LLM_SEMAPHORE = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)


def get_llm() -> ChatGroq:
    """Return the Groq chat model used across the app."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0,
    )


def get_structured_llm():
    """Return the Groq model bound to the WeatherIntent schema."""
    llm = get_llm()
    return llm.with_structured_output(WeatherIntent)


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content)


def _is_retryable_llm_error(error: Exception) -> bool:
    message = str(error).lower()
    retry_markers = (
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "timeout",
        "connection",
        "429",
        "503",
    )
    return any(marker in message for marker in retry_markers)


async def _invoke_with_retries(model: Any, prompt: Any) -> Any:
    last_error: Exception | None = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            async with _LLM_SEMAPHORE:
                return await asyncio.wait_for(
                    model.ainvoke(prompt),
                    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                )
        except asyncio.TimeoutError as exc:
            last_error = TimeoutError(
                f"LLM request timed out after {LLM_REQUEST_TIMEOUT_SECONDS:.0f}s."
            )
            if attempt >= LLM_MAX_RETRIES:
                raise last_error from exc
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_RETRIES or not _is_retryable_llm_error(exc):
                raise
            await asyncio.sleep(LLM_RETRY_BACKOFF_SECONDS * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM invocation failed unexpectedly.")


async def call_llm(prompt: Any) -> str:
    """Call Groq and return the response text."""
    llm = get_llm()
    response = await _invoke_with_retries(llm, prompt)
    return _extract_text(response)


async def call_structured_llm(prompt: Any) -> WeatherIntent:
    """Call Groq with structured output for weather intent parsing."""
    structured_llm = get_structured_llm()
    return await _invoke_with_retries(structured_llm, prompt)
