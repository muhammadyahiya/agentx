"""Provider subsystem: a registry + factory for building LLM clients."""
from .base import ProviderError, ProviderSpec
from .factory import get_chat_model, get_crewai_llm, list_providers
from .registry import all_specs, canonical_ids, get_spec

__all__ = [
    "ProviderError",
    "ProviderSpec",
    "get_chat_model",
    "get_crewai_llm",
    "list_providers",
    "all_specs",
    "canonical_ids",
    "get_spec",
]
