"""Tree-sitter RAG retriever system.

This module provides functionality to store and retrieve code chunks 
using semantic similarity search.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
import math
import subprocess

from .chunker import TreeSitterChunker
from .models import CodeChunk, RetrievalResult
from .document_store import SimpleDocumentStore
from .embeddings import EmbeddingSystem
from .memory_manager import MemoryManager

SUPPORTED_EXTENSIONS = {
    ".py",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    "target",
}

MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB


class PersistentRetriever:
    """Persistent retriever for code chunks using SQLite storage.
    """
    
    def __init__(self, db_path: str = "rag_index.db"):
        self.store = SimpleDocumentStore(db_path)
        self.embedding_system = EmbeddingSystem(db_path)
        self.memory_manager = MemoryManager(db_path)
    
    def add_chunks_with_embeddings(self, chunks: List[CodeChunk]) -> None:
        """Add chunks to the retriever and generate embeddings for them.
        
        Args:
            chunks: List of CodeChunks to add
        """
        self.embedding_system.add_chunks_with_embeddings(chunks)
    
    def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add chunks to the retriever."""
        self.store.add_chunks(chunks)
    
    def add_chunk(self, chunk: CodeChunk) -> None:
        """Add a single chunk to the retriever."""
        self.store.add_chunk(chunk)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk by ID."""
        return self.store.remove_chunk(chunk_id)
    
    def get_chunks_by_session(self, session_id: str) -> List[CodeChunk]:
        """Get all chunks associated with a session."""
        return self.memory_manager.get_chunks_by_session(session_id)
    
    def search(self, query: str, top_k: int = 5, hybrid_alpha: float = 0.5, session_id: Optional[str] = None) -> List[RetrievalResult]:
        """Search for relevant chunks based on a query using hybrid retrieval.
        
        This implementation uses both BM25 (text-based) and semantic similarity search,
        combining them with a weighted approach.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            hybrid_alpha: Weight for semantic search (0.0 = BM25 only, 1.0 = semantic only)
            session_id: Optional session identifier for memory tracking
            
        Returns:
            List of retrieval results with scores
        """
        # Get all chunks from the store
        all_chunks = self.store.get_all_chunks()
        
        if not all_chunks:
            return []
        
        # Check if we have embeddings for any chunks
        has_embeddings = False
        for chunk in all_chunks:
            if self.embedding_system.embedding_store.get_embedding(chunk.id) is not None:
                has_embeddings = True
                break
        
        # If no embeddings available, fall back to BM25 only
        if not has_embeddings:
            print("Warning: No embeddings found. Using BM25-only search.")
            results = self._bm25_search(query, top_k)
            
            # Track session memory if session_id is provided
            if session_id:
                for result in results:
                    self.memory_manager.track_session_chunk(session_id, result.chunk.id)
            
            return results
        
        # Use hybrid search with both BM25 and semantic similarity
        results = self._hybrid_search(query, top_k, hybrid_alpha)
        
        # Track session memory if session_id is provided
        if session_id:
            for result in results:
                self.memory_manager.track_session_chunk(session_id, result.chunk.id)
        
        return results
    
    def _bm25_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Perform BM25-based search using the document store's search method."""
        # This uses the existing search implementation in DocumentStore
        return self.store.search(query, top_k)
    
    def _hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[RetrievalResult]:
        """Perform hybrid search combining BM25 and semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            alpha: Weight for semantic search (0.0 = BM25 only, 1.0 = semantic only)
            
        Returns:
            List of retrieval results with combined scores
        """
        # Get all chunks from the store
        all_chunks = self.store.get_all_chunks()
        
        if not all_chunks:
            return []
        
        # Get BM25 scores for all chunks
        bm25_results = self._bm25_search(query, len(all_chunks))
        bm25_scores = {result.chunk.id: result.score for result in bm25_results}
        
        # Get semantic scores for all chunks that have embeddings
        semantic_results = []
        try:
            semantic_results = self.embedding_system.search_with_embeddings(query, len(all_chunks))
        except Exception as e:
            print(f"Warning: Semantic search failed: {e}")
            # Fall back to BM25 only if semantic fails
            return self._bm25_search(query, top_k)
        
        # Create a mapping of chunk_id -> semantic_score
        semantic_scores = {}
        for chunk, score in semantic_results:
            semantic_scores[chunk.id] = score
        
        # Combine scores using hybrid approach with proper normalization
        combined_scores = []
        for chunk in all_chunks:
            bm25_score = bm25_scores.get(chunk.id, 0.0)
            semantic_score = semantic_scores.get(chunk.id, 0.0)
            
            # Normalize BM25 score to [0, 1] range if needed (though DocumentStore already does this)
            # Combine scores with alpha weight
            combined_score = alpha * semantic_score + (1 - alpha) * bm25_score
            
            combined_scores.append((chunk, combined_score))
        
        # Sort by combined score and return top_k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(chunk=chunk, score=score) 
            for chunk, score in combined_scores[:top_k]
        ]
    
    def _normalize_bm25_score(self, score: float) -> float:
        """Normalize BM25 score to [0, 1] range.
        
        Note: In the current implementation, BM25 scores are already normalized 
        by the DocumentStore.search() method, so this is a no-op.
        """
        # Since BM25 scores are already in [0, 1] range from the search implementation,
        # we just return the score as-is
        return score
    
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a specific chunk by ID."""
        return self.store.get_chunk(chunk_id)
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks from a specific file."""
        return self.store.get_chunks_by_file(file_path)
    
    def get_all_chunks(self) -> List[CodeChunk]:
        """Get all chunks in the retriever."""
        return self.store.get_all_chunks()
    
    def size(self) -> int:
        """Get the number of chunks in the retriever."""
        return self.store.size()


class RAGSystem:
    """Main RAG system that coordinates chunking and retrieval."""
    
    def __init__(self, db_path: str = "rag_index.db"):
        self.chunker = TreeSitterChunker()
        self.retriever = PersistentRetriever(db_path)

    
    def index_repository(
        self,
        repo_root: Path,
        *,
        batch_size: int = 64,
    ) -> None:
        """Index repository source files in bounded batches."""

        repo_root = Path(repo_root).resolve()

        batch: list[CodeChunk] = []
        total_chunks = 0
        total_files = 0

        for file_path in self._iter_source_files(repo_root):
            try:
                file_chunks = self.chunker.chunk_file(
                    file_path,
                    repo_root,
                )

            except Exception as exc:
                print(
                    f"Warning: Could not process "
                    f"{file_path}: {exc}"
                )
                continue

            total_files += 1

            for chunk in file_chunks:
                batch.append(chunk)

                if len(batch) >= batch_size:
                    self.retriever.add_chunks_with_embeddings(
                        batch
                    )

                    total_chunks += len(batch)

                    print(
                        f"Indexed {total_chunks} chunks "
                        f"from {total_files} files..."
                    )

                    batch.clear()

        # Flush anything left over.
        if batch:
            self.retriever.add_chunks_with_embeddings(
                batch
            )

            total_chunks += len(batch)
            batch.clear()

        print(
            f"Indexed {total_chunks} code chunks "
            f"from {total_files} files in {repo_root}"
        ) 

    def get_all_chunks(self) -> List[CodeChunk]:
        """Return all indexed code chunks."""
        return self.retriever.get_all_chunks()

    def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.5,
        session_id: str | None = None,
    ) -> list[RetrievalResult]:

        return self.retriever.search(
            query=query,
            top_k=top_k,
            hybrid_alpha=hybrid_alpha,
            session_id=session_id,
        ) 

    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a specific chunk by ID."""
        return self.retriever.get_chunk(chunk_id)
    
    def size(self) -> int:
        """Get the number of chunks in the retriever."""
        return self.retriever.size()

    def _iter_source_files(
        self,
        repo_root: Path,
    ):
        """Yield source files suitable for RAG indexing."""

        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            relative_paths = [
                path
                for path in result.stdout.split("\0")
                if path
            ]

            candidates = (
                repo_root / relative
                for relative in relative_paths
            )

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Allow RAGSystem to still work outside Git repositories.
            candidates = repo_root.rglob("*")

        for file_path in candidates:
            if not file_path.is_file():
                continue

            try:
                relative = file_path.relative_to(repo_root)
            except ValueError:
                continue

            if any(
                part in EXCLUDED_DIRS
                for part in relative.parts[:-1]
            ):
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > MAX_FILE_BYTES:
                print(
                    f"Skipping large file: {relative} "
                    f"({size / (1024 * 1024):.1f} MiB)"
                )
                continue

            yield file_path