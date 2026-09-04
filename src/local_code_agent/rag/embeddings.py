"""Tree-sitter RAG embedding system.

This module provides functionality to generate and manage embeddings for code chunks
using Ollama's embedding capabilities.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple
import sqlite3
from dataclasses import dataclass, asdict

from .models import CodeChunk
from .document_store import DocumentStore, DocumentStoreConfig

from ollama import Client

from local_code_agent.config import settings

@dataclass
class Embedding:
    """Represents an embedding vector for a code chunk."""
    chunk_id: str
    vector: List[float]
    created_at: str


class EmbeddingStore:
    """Persistent store for embeddings using SQLite."""
    
    def __init__(self, db_path: str = "rag_index.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database for embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            # Create table for embeddings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """)
            
            # Create index for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id)")
    
    def add_embedding(self, chunk_id: str, vector: List[float]) -> None:
        """Add an embedding for a chunk."""
        with sqlite3.connect(self.db_path) as conn:
            # Convert vector to bytes for storage
            vector_bytes = self._vector_to_bytes(vector)
            conn.execute("""
                INSERT OR REPLACE INTO embeddings 
                (chunk_id, vector, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (chunk_id, vector_bytes))
    
    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Get embedding for a chunk."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT vector FROM embeddings WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            
            if row:
                return self._bytes_to_vector(row[0])
            return None
    
    def get_all_embeddings(self) -> List[Tuple[str, List[float]]]:
        """Get all embeddings in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT chunk_id, vector FROM embeddings")
            rows = cursor.fetchall()
            
            return [(row[0], self._bytes_to_vector(row[1])) for row in rows]
    
    def _vector_to_bytes(self, vector: List[float]) -> bytes:
        """Convert a list of floats to bytes for storage."""
        import struct
        return struct.pack(f'{len(vector)}f', *vector)
    
    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """Convert bytes back to a list of floats."""
        import struct
        # Calculate number of floats from the byte length
        num_floats = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{num_floats}f', data))



class EmbeddingSystem:
    def __init__(
        self,
        db_path: str = "rag_index.db",
        *,
        ollama_host: str | None = None,
        embedding_model: str | None = None,
    ):
        self.db_path = db_path

        # Use explicit values when supplied, otherwise inherit the
        # same configuration used by the main local-code-agent.
        self.ollama_host = ollama_host or settings.ollama_host
        self.embedding_model = (
            embedding_model or settings.ollama_embedding_model
        )

        self.client = Client(
            host=self.ollama_host,
        )

        self.embedding_store = EmbeddingStore(db_path)

        self.document_store = DocumentStore(
            DocumentStoreConfig(
                db_path=db_path,
            )
        )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using the configured Ollama server.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            RuntimeError: If Ollama cannot generate an embedding.
        """

        try:
            response = self.client.embed(
                model=self.embedding_model,
                input=text,
            )

            if not response.embeddings:
                raise RuntimeError(
                    "Ollama returned no embedding vectors."
                )

            return list(response.embeddings[0])

        except Exception as exc:
            raise RuntimeError(
                "Failed to generate embedding "
                f"using model '{self.embedding_model}' "
                f"at '{self.ollama_host}': {exc}"
            ) from exc  

    def generate_chunk_embeddings(self, chunks: List[CodeChunk]) -> List[Tuple[CodeChunk, List[float]]]:
        """Generate embeddings for a list of code chunks.
        
        Args:
            chunks: List of CodeChunks to embed
            
        Returns:
            List of tuples containing (chunk, embedding_vector)
        """
        results = []
        for chunk in chunks:
            # Create a meaningful text representation for the embedding
            embedding_text = self._create_embedding_text(chunk)
            vector = self.generate_embedding(embedding_text)
            results.append((chunk, vector))
        
        return results
    
    def _create_embedding_text(self, chunk: CodeChunk) -> str:
        """Create a text representation suitable for embedding generation.
        
        This combines the chunk's content with metadata to create rich context
        for better embeddings.
        """
        # Create a structured text that includes both code and context
        parts = [
            f"Language: {chunk.language}",
            f"Type: {chunk.node_type}",
            f"Symbol: {chunk.symbol or 'N/A'}",
            f"Parent: {chunk.parent_symbol or 'N/A'}",
            f"Content:\n{chunk.content}"
        ]
        
        return "\n\n".join(parts)
    
    def add_chunks_with_embeddings(self, chunks: List[CodeChunk]) -> None:
        """Add chunks to the document store and generate embeddings for them.
        
        Args:
            chunks: List of CodeChunks to add
        """
        # First add chunks to the document store
        self.document_store.add_chunks(chunks)
        
        # Then generate embeddings for all chunks
        chunk_embeddings = self.generate_chunk_embeddings(chunks)
        
        # Store the embeddings
        for chunk, vector in chunk_embeddings:
            self.embedding_store.add_embedding(chunk.id, vector)
    
    def search_with_embeddings(self, query: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Search for relevant chunks using embedding similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples containing (chunk, similarity_score)
        """
        # Generate embedding for the query
        query_vector = self.generate_embedding(query)
        
        # Get all chunks from the document store
        all_chunks = self.document_store.get_all_chunks()
        
        # Calculate similarities with query vector
        similarities = []
        for chunk in all_chunks:
            # Get the stored embedding for this chunk
            chunk_vector = self.embedding_store.get_embedding(chunk.id)
            
            if chunk_vector is not None:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, chunk_vector)
                similarities.append((chunk, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        import math
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        # Handle edge case where one vector is zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        return dot_product / (magnitude1 * magnitude2)