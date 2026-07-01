"""A small, swappable RAG pipeline.

Strategy:
  * Split text into chunks (LangChain splitter if available, else a simple splitter).
  * Index with Chroma + embeddings when ``[rag]`` is installed.
  * Otherwise fall back to an in-memory keyword retriever (no deps), so the
    generated project runs immediately and can be upgraded later.

Embedding providers are resolved in this order when no explicit config is given:
  1. Provider specified via ``AGENTX_EMBEDDING_PROVIDER`` env var.
  2. HuggingFace local (no API key, best offline default).
  3. OpenAI (if ``OPENAI_API_KEY`` is set).
  4. Ollama (if the package is installed).
  5. Keyword-only fallback.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from ..config import get_settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Text splitting
# ──────────────────────────────────────────────────────────────────────────────

def _split(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return [c for c in splitter.split_text(text) if c.strip()]
    except ImportError:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start : start + chunk_size])
            start += chunk_size - overlap
        return [c for c in chunks if c.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Keyword retrieval helpers
# ──────────────────────────────────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return [t for t in _TOKEN.findall(s.lower()) if len(t) > 1]


# ──────────────────────────────────────────────────────────────────────────────
# RagIndex
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RagIndex:
    """Holds chunks and answers similarity queries.

    If a vector store is available it is used; otherwise a keyword scorer ranks
    chunks. The public ``search`` API is identical either way.
    """

    chunks: list[str] = field(default_factory=list)
    _store: object | None = None

    def search(self, query: str, k: int = 4) -> list[str]:
        if self._store is not None:
            try:
                docs = self._store.similarity_search(query, k=k)
                return [d.page_content for d in docs]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Vector search failed, falling back to keyword retrieval: %s", exc
                )
        if not self.chunks:
            return []
        q = Counter(_tokens(query))
        scored = [(sum(_tokens(c).count(t) for t in q), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for s, c in scored if s > 0][:k]
        return top or self.chunks[:k]

    def context(self, query: str, k: int = 4) -> str:
        return "\n---\n".join(self.search(query, k))


# ──────────────────────────────────────────────────────────────────────────────
# Index builder
# ──────────────────────────────────────────────────────────────────────────────

def build_index_from_texts(
    texts: list[str],
    persist_dir: str | None = None,
    embeddings=None,
    embedding_config=None,
) -> RagIndex:
    """Build a ``RagIndex`` from raw texts.

    Args:
        texts: List of documents / raw text strings to index.
        persist_dir: Optional directory to persist the Chroma store on disk.
        embeddings: A pre-built LangChain ``Embeddings`` instance. Takes
            priority over ``embedding_config``.
        embedding_config: An ``EmbeddingConfig`` subclass (e.g.
            ``HuggingFaceEmbeddingConfig``). Resolved to an ``Embeddings``
            object via ``get_embeddings()``. Falls back to auto-detection
            when both ``embeddings`` and ``embedding_config`` are ``None``.

    Returns:
        A ``RagIndex`` backed by Chroma (if available) or keyword retrieval.
    """
    chunks: list[str] = []
    for t in texts:
        chunks.extend(_split(t))
    logger.debug("Split %d documents into %d chunks", len(texts), len(chunks))

    store = None
    try:
        from langchain_chroma import Chroma  # type: ignore

        if embeddings is None:
            from .embeddings import get_embeddings  # lazy to avoid circular import
            embeddings = get_embeddings(embedding_config or _settings_embedding_config())

        if embeddings is not None:
            logger.info(
                "Building Chroma index: %d chunks, provider=%s",
                len(chunks),
                type(embeddings).__name__,
            )
            store = Chroma.from_texts(
                chunks, embedding=embeddings, persist_directory=persist_dir
            )
        else:
            logger.info(
                "No embeddings available; using in-memory keyword retriever for %d chunks",
                len(chunks),
            )
    except ImportError:
        logger.info(
            "Chroma not installed; using in-memory keyword retriever. "
            "Install 'agentx-kit[rag]' to enable vector search."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Vector index build failed (%s); falling back to keyword retriever.", exc
        )

    return RagIndex(chunks=chunks, _store=store)


def _settings_embedding_config():
    """Read ``AGENTX_EMBEDDING_PROVIDER`` / ``AGENTX_EMBEDDING_MODEL`` from settings.

    Returns a typed ``EmbeddingConfig`` subclass when the provider is configured,
    else ``None`` (triggers ``auto_embeddings()``).
    """
    from .embeddings import (  # lazy import
        AzureOpenAIEmbeddingConfig,
        BedrockEmbeddingConfig,
        CohereEmbeddingConfig,
        GoogleEmbeddingConfig,
        HuggingFaceEmbeddingConfig,
        OllamaEmbeddingConfig,
        OpenAIEmbeddingConfig,
        VoyageEmbeddingConfig,
    )

    s = get_settings()
    provider = s.default_embedding_provider.strip().lower()
    model = s.default_embedding_model.strip() or None

    _MAP = {
        "huggingface": HuggingFaceEmbeddingConfig,
        "hf": HuggingFaceEmbeddingConfig,
        "openai": OpenAIEmbeddingConfig,
        "azure": AzureOpenAIEmbeddingConfig,
        "cohere": CohereEmbeddingConfig,
        "google": GoogleEmbeddingConfig,
        "bedrock": BedrockEmbeddingConfig,
        "aws": BedrockEmbeddingConfig,
        "voyage": VoyageEmbeddingConfig,
        "ollama": OllamaEmbeddingConfig,
    }

    cls = _MAP.get(provider)
    if cls is None:
        return None

    kwargs = {}
    if model:
        kwargs["model"] = model
    logger.debug(
        "Settings-based embedding config: provider=%s model=%s", provider, model or "(default)"
    )
    return cls(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# LangChain tool adapter
# ──────────────────────────────────────────────────────────────────────────────

def make_retriever_tool(index: RagIndex):
    """Expose a ``RagIndex`` as a LangChain retrieval ``@tool``."""
    from langchain_core.tools import tool

    @tool
    def knowledge_base(query: str) -> str:
        """Search the project's knowledge base and return relevant passages."""
        return index.context(query)

    return knowledge_base
