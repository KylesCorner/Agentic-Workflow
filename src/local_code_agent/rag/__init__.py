"""Tree-sitter RAG subsystem.

This module provides Tree-sitter based semantic chunking and retrieval
for code understanding in the local code agent.

Modules:
- chunker: Tree-sitter based code chunking
- languages: Language configurations for Tree-sitter
- models: Data models for code chunks
- retriever: Retrieval system for code chunks
- document_store: Persistent storage for code chunks
- embeddings: Embedding generation and similarity search
"""

from .chunker import TreeSitterChunker
from .languages import language_for_path, parser_for
from .models import CodeChunk, RetrievalResult
from .retriever import RAGSystem, PersistentRetriever 
from .document_store import DocumentStore, SimpleDocumentStore
from .embeddings import EmbeddingSystem, EmbeddingStore

__all__ = [
    "TreeSitterChunker",
    "language_for_path", 
    "parser_for",
    "CodeChunk",
    "RAGSystem",
    "PersistentRetriever",
    "RetrievalResult",
    "DocumentStore",
    "SimpleDocumentStore",
    "EmbeddingSystem",
    "EmbeddingStore"
]
