"""Memory management system for persistent RAG memory.

This module provides enhanced memory management capabilities including:
- Session-based memory tracking
- Memory eviction policies (LRU, LFU)
- Access statistics and monitoring
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import sqlite3

from .document_store import DocumentStore, DocumentStoreConfig
from .models import CodeChunk

logger = logging.getLogger(__name__)


class MemoryManager:
    """Enhanced memory manager for RAG system with session tracking and eviction policies."""
    
    # def __init__(self, db_path: str = "rag_index.db"):
    #     self.db_path = Path(db_path)
    #     self.document_store = DocumentStore(
    #         DocumentStoreConfig(db_path=db_path)
    #     )
    #     self._session_memory: Dict[str, List[str]] = {}  # session_id -> list of chunk_ids
    def __init__(self, db_path: str = "rag_index.db"):
        self.db_path = Path(db_path)

        self.document_store = DocumentStore(
            DocumentStoreConfig(
                db_path=str(self.db_path),
            )
        )

        self._init_session_tables()

    def _init_session_tables(self) -> None:
        """Create persistent RAG session tracking tables."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_chunks (
                    session_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,

                    first_accessed TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,

                    last_accessed TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,

                    access_count INTEGER
                        NOT NULL DEFAULT 1,

                    PRIMARY KEY (
                        session_id,
                        chunk_id
                    )
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_session_chunks_session
                ON session_chunks(session_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_session_chunks_last_accessed
                ON session_chunks(last_accessed)
                """
            ) 

    def track_session_chunk(
        self,
        session_id: str,
        chunk_id: str,
    ) -> None:
        """Persist that a session retrieved a chunk."""

        if not session_id:
            return

        # Update global chunk statistics.
        self.document_store.update_access_info(
            chunk_id
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO session_chunks (
                    session_id,
                    chunk_id,
                    first_accessed,
                    last_accessed,
                    access_count
                )
                SELECT
                    ?,
                    id,
                    CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP,
                    1
                FROM chunks
                WHERE id = ?

                ON CONFLICT(session_id, chunk_id)
                DO UPDATE SET
                    last_accessed = CURRENT_TIMESTAMP,
                    access_count =
                        session_chunks.access_count + 1
                """,
                (
                    session_id,
                    chunk_id,
                ),
            ) 

    def get_session_chunks(
        self,
        session_id: str,
    ) -> list[str]:
        """Return chunk IDs previously used by a session."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT chunk_id
                FROM session_chunks
                WHERE session_id = ?
                ORDER BY last_accessed DESC
                """,
                (session_id,),
            )

            return [
                row[0]
                for row in cursor.fetchall()
            ] 

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """Forget retrieval history for one session."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                DELETE FROM session_chunks
                WHERE session_id = ?
                """,
                (session_id,),
            ) 

    def get_session_stats(
        self,
        session_id: str,
    ) -> dict[str, int]:
        """Return persistent retrieval stats for a session."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(access_count), 0)
                FROM session_chunks
                WHERE session_id = ?
                """,
                (session_id,),
            )

            unique_chunks, total_accesses = cursor.fetchone()

        return {
            "unique_chunks": unique_chunks,
            "total_accesses": total_accesses,
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = self.document_store.get_memory_stats()
        stats['active_sessions'] = len(self._session_memory)
        return stats
    
    def evict_least_used_chunks(self, max_chunks: int = 1000) -> int:
        """Evict the least used chunks to maintain memory limits.
        
        Args:
            max_chunks: Maximum number of chunks to keep
            
        Returns:
            Number of chunks evicted
        """
        current_count = self.document_store.size()
        
        if current_count <= max_chunks:
            return 0
        
        # Get least accessed chunks
        least_used = self.document_store.get_least_accessed_chunks(current_count - max_chunks)
        
        # Remove them from database.
        deleted_count = 0
        for chunk in least_used:
            if self.document_store.remove_chunk(chunk.id):
                deleted_count += 1
        
        logger.info(f"Evicted {deleted_count} least used chunks to maintain limit of {max_chunks}")
        return deleted_count
    
    def add_tags_to_chunk(self, chunk_id: str, tags: List[str]) -> None:
        """Add tags to a specific chunk."""
        chunk = self.document_store.get_chunk(chunk_id)
        if chunk:
            chunk.tags.extend(tag for tag in tags if tag not in chunk.tags)
            chunk.updated_at = datetime.now()
            self.document_store.add_chunk(chunk)
    
    def get_chunks_by_session(self, session_id: str) -> List[CodeChunk]:
        """Get all chunks associated with a session."""
        chunk_ids = self.get_session_chunks(session_id)
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.document_store.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def get_recently_accessed(self, limit: int = 20) -> List[CodeChunk]:
        """Get most recently accessed chunks."""
        with sqlite3.connect(self.document_store.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM chunks
                ORDER BY last_accessed DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [self.document_store._row_to_chunk(row) for row in rows]