"""Structured outputs — typed, validated model responses.

Thin, provider-agnostic wrappers over LangChain's ``with_structured_output``
(supported across OpenAI, Anthropic, Gemini, Bedrock, Groq, Ollama, …). Pass a
Pydantic model or a JSON schema; get back validated objects instead of strings.
"""
from __future__ import annotations

from typing import Any

from .providers import get_chat_model


def structured_model(schema: Any, provider: str | None = None, model: str | None = None, **kwargs):
    """Return a chat model that emits instances of ``schema`` (a Pydantic model).

    Example::

        from pydantic import BaseModel
        class Person(BaseModel):
            name: str
            age: int
        llm = structured_model(Person, "openai", "gpt-4o-mini")
        person = llm.invoke("Alice is 30")   # -> Person(name='Alice', age=30)
    """
    llm = get_chat_model(provider, model, **kwargs)
    return as_structured(llm, schema)


def as_structured(llm: Any, schema: Any):
    """Attach structured-output decoding to an existing chat model."""
    if not hasattr(llm, "with_structured_output"):
        raise TypeError(
            f"{type(llm).__name__} does not support structured output; "
            "use a provider/model that implements with_structured_output()."
        )
    return llm.with_structured_output(schema)
