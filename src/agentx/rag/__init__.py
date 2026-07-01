"""RAG: chunk documents, index them, retrieve relevant passages.

Uses Chroma when ``agentx-kit[rag]`` is installed; otherwise falls back to a
dependency-free in-memory keyword retriever so RAG works out of the box.

Embedding providers (``EmbeddingConfig`` subclasses) let you choose the vector
representation.  Install the matching extra and pass the config to
``build_index_from_texts(embedding_config=...)``.
"""
from .embeddings import (
    AnyEmbeddingConfig,
    AzureOpenAIEmbeddingConfig,
    BedrockEmbeddingConfig,
    CohereEmbeddingConfig,
    EmbeddingConfig,
    GoogleEmbeddingConfig,
    HuggingFaceEmbeddingConfig,
    OllamaEmbeddingConfig,
    OpenAIEmbeddingConfig,
    VoyageEmbeddingConfig,
    auto_embeddings,
    get_embeddings,
)
from .pipeline import RagIndex, build_index_from_texts, make_retriever_tool

__all__ = [
    # Pipeline
    "RagIndex",
    "build_index_from_texts",
    "make_retriever_tool",
    # Embedding factory
    "get_embeddings",
    "auto_embeddings",
    # Embedding configs
    "EmbeddingConfig",
    "AnyEmbeddingConfig",
    "HuggingFaceEmbeddingConfig",
    "OpenAIEmbeddingConfig",
    "AzureOpenAIEmbeddingConfig",
    "CohereEmbeddingConfig",
    "GoogleEmbeddingConfig",
    "BedrockEmbeddingConfig",
    "VoyageEmbeddingConfig",
    "OllamaEmbeddingConfig",
]
