"""Tree-sitter RAG document store for persistent chunk storage.

This module provides a persistent storage system for code chunks extracted 
by the Tree-sitter RAG system.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import sqlite3
import logging

from .models import CodeChunk, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class DocumentStoreConfig:
    """Configuration for the document store."""
    db_path: str = "rag_index.db"
    auto_commit: bool = True


class DocumentStore:
    """Persistent document store for code chunks.
    
    This store uses SQLite to persist code chunks and provides efficient 
    retrieval capabilities.
    """
    
    def __init__(self, config: DocumentStoreConfig | None = None):
        self.config = config or DocumentStoreConfig()
        self.db_path = Path(self.config.db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    start_byte INTEGER NOT NULL,
                    end_byte INTEGER NOT NULL,
                    node_type TEXT NOT NULL,
                    symbol TEXT,
                    parent_symbol TEXT,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_node_type ON chunks(node_type)")
            
            # Create a table for tracking repository indexing
            conn.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0
                )
            """)
    
    def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add multiple chunks to the store."""
        if not chunks:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            for chunk in chunks:
                # Convert dataclass to dict for database storage
                chunk_dict = asdict(chunk)
                
                # Remove the id field from the dict since we'll use it as primary key
                chunk_id = chunk_dict.pop('id')
                
                # Insert or update the chunk
                conn.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, path, language, start_line, end_line, start_byte, end_byte, 
                     node_type, symbol, parent_symbol, content, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    chunk_id,
                    chunk_dict['path'],
                    chunk_dict['language'],
                    chunk_dict['start_line'],
                    chunk_dict['end_line'],
                    chunk_dict['start_byte'],
                    chunk_dict['end_byte'],
                    chunk_dict['node_type'],
                    chunk_dict.get('symbol'),
                    chunk_dict.get('parent_symbol'),
                    chunk_dict['content']
                ))
    
    def add_chunk(self, chunk: CodeChunk) -> None:
        """Add a single chunk to the store."""
        self.add_chunks([chunk])
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            return cursor.rowcount > 0
    
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a specific chunk by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_chunk(row)
            return None
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks from a specific file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM chunks WHERE path = ? ORDER BY start_line", (file_path,))
            rows = cursor.fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    def get_chunks_by_language(self, language: str) -> List[CodeChunk]:
        """Get all chunks of a specific language."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM chunks WHERE language = ? ORDER BY start_line", (language,))
            rows = cursor.fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for relevant chunks based on a query.
        
        This uses SQLite's full-text search capabilities for better performance.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of retrieval results with scores
        """
        # Simple scoring based on keyword matching in content and symbol names
        # In a more advanced system, this would use vector embeddings
        query_lower = query.lower()
        
        with sqlite3.connect(self.db_path) as conn:
            # Use SQLite's LIKE operator for pattern matching
            cursor = conn.execute("""
                SELECT *, 
                       (CASE WHEN content LIKE ? THEN 0.5 ELSE 0 END +
                        CASE WHEN symbol LIKE ? THEN 0.3 ELSE 0 END +
                        CASE WHEN parent_symbol LIKE ? THEN 0.2 ELSE 0 END +
                        CASE WHEN language LIKE ? THEN 0.1 ELSE 0 END) as score
                FROM chunks 
                WHERE content LIKE ? OR symbol LIKE ? OR parent_symbol LIKE ?
                ORDER BY score DESC
                LIMIT ?
            """, (f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%", 
                  f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%", 
                  f"%{query_lower}%", top_k))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                chunk = self._row_to_chunk(row)
                score = float(row[-1])  # The last column is the calculated score
                # Normalize score to [0, 1] range if needed
                if score > 1.0:
                    score = 1.0
                results.append(RetrievalResult(chunk=chunk, score=score))
            
            return results
    
    def get_all_chunks(self) -> List[CodeChunk]:
        """Get all chunks in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM chunks ORDER BY start_line")
            rows = cursor.fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    def size(self) -> int:
        """Get the number of chunks in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
    
    def get_repository_info(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a repository's indexing status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM repositories WHERE path = ?", (repo_path,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'path': row[1],
                    'indexed_at': row[2],
                    'chunk_count': row[3]
                }
            return None
    
    def mark_repository_indexed(self, repo_path: str, chunk_count: int = 0) -> None:
        """Mark a repository as indexed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO repositories (path, chunk_count)
                VALUES (?, ?)
            """, (repo_path, chunk_count))
    
    def _row_to_chunk(self, row: tuple) -> CodeChunk:
        """Convert a database row to a CodeChunk object."""
        # Row structure from chunks table:
        # id, path, language, start_line, end_line, start_byte, end_byte, 
        # node_type, symbol, parent_symbol, content, created_at, updated_at
        
        return CodeChunk(
            id=row[0],
            path=row[1],
            language=row[2],
            start_line=row[3],
            end_line=row[4],
            start_byte=row[5],
            end_byte=row[6],
            node_type=row[7],
            symbol=row[8],
            content=row[10],  # Content is at index 10
            parent_symbol=row[9]  # Parent symbol is at index 9
        )
    
    def clear(self) -> None:
        """Clear all data from the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM repositories")


class SimpleDocumentStore:
    """A simplified document store that works with the existing RAG system.
    
    This is a wrapper around the full DocumentStore to maintain compatibility
    with the existing interface while providing persistent storage.
    """
    
    def __init__(self, db_path: str = "rag_index.db"):
        self.store = DocumentStore(DocumentStoreConfig(db_path=db_path))
    
    def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add chunks to the store."""
        self.store.add_chunks(chunks)
    
    def add_chunk(self, chunk: CodeChunk) -> None:
        """Add a single chunk to the store."""
        self.store.add_chunk(chunk)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk by ID."""
        return self.store.remove_chunk(chunk_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a specific chunk by ID."""
        return self.store.get_chunk(chunk_id)
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for relevant chunks based on a query."""
        return self.store.search(query, top_k)
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks from a specific file."""
        return self.store.get_chunks_by_file(file_path)
    
    def get_all_chunks(self) -> List[CodeChunk]:
        """Get all chunks in the store."""
        return self.store.get_all_chunks()
    
    def size(self) -> int:
        """Get the number of chunks in the store."""
        return self.store.size()
    
    def clear(self) -> None:
        """Clear all data from the store."""
        self.store.clear()