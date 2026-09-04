"""Tree-sitter RAG tool for code understanding and retrieval.

This module provides tools to integrate Tree-sitter based RAG (Retrieval-Augmented Generation)
into the local code agent's toolset, enabling semantic code search and understanding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from local_code_agent.rag.retriever import RAGSystem
from local_code_agent.tools.registry import ToolSpec


class RAGTools:
    """Tools for interacting with the Tree-sitter RAG system."""
    
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()
        # Initialize the RAG system with a database in the repository
        self.rag_system = RAGSystem(db_path=str(self.repo / ".rag_index.db"))
        
    def index_repository(self, force: bool = False) -> str:
        """Index all code files in the repository for semantic search.
        
        Args:
            force: If True, re-index even if already indexed
            
        Returns:
            Status message about indexing
        """
        try:
            # Check if already indexed
            if not force and (self.repo / ".rag_indexed").exists():
                return "Repository already indexed. Use force=True to re-index."
            
            self.rag_system.index_repository(self.repo)
            
            # Mark as indexed
            (self.repo / ".rag_indexed").touch()
            
            size = self.rag_system.size()
            return f"Successfully indexed repository with {size} code chunks."
        except Exception as e:
            return f"Error indexing repository: {e}"
    
    def search_code(self, query: str, top_k: int = 5, hybrid_alpha: float = 0.5) -> str:
        """Search for relevant code chunks using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return (default: 5)
            hybrid_alpha: Weight for semantic search (0.0 = BM25 only, 1.0 = semantic only)
            
        Returns:
            Formatted search results
        """
        try:
            results = self.rag_system.search(query, top_k=top_k, hybrid_alpha=hybrid_alpha)
            
            if not results:
                return "No relevant code chunks found."
            
            output = f"Found {len(results)} relevant code chunks:\n\n"
            
            for i, result in enumerate(results, 1):
                chunk = result.chunk
                score = result.score
                
                output += f"{i}. {chunk.symbol or 'N/A'} ({chunk.path}:{chunk.start_line}-{chunk.end_line})\n"
                output += f"   Score: {score:.3f}\n"
                output += f"   Type: {chunk.node_type}\n"
                output += f"   Language: {chunk.language}\n"
                output += f"   Content:\n{chunk.content[:200]}...\n\n"
            
            return output
        except Exception as e:
            return f"Error searching code: {e}"
    
    def get_chunk_details(self, chunk_id: str) -> str:
        """Get detailed information about a specific code chunk.
        
        Args:
            chunk_id: ID of the code chunk
            
        Returns:
            Formatted chunk details
        """
        try:
            chunk = self.rag_system.get_chunk(chunk_id)
            if not chunk:
                return f"Chunk with ID '{chunk_id}' not found."
            
            output = f"Chunk Details:\n"
            output += f"ID: {chunk.id}\n"
            output += f"File: {chunk.path}\n"
            output += f"Line: {chunk.start_line}-{chunk.end_line}\n"
            output += f"Type: {chunk.node_type}\n"
            output += f"Symbol: {chunk.symbol or 'N/A'}\n"
            output += f"Parent Symbol: {chunk.parent_symbol or 'N/A'}\n"
            output += f"Language: {chunk.language}\n"
            output += f"Content:\n{chunk.content}\n"
            
            return output
        except Exception as e:
            return f"Error getting chunk details: {e}"


def create_rag_tool(repo: Path) -> ToolSpec:
    """Create a RAG tool specification for the tool registry.
    
    Args:
        repo: Repository path
        
    Returns:
        ToolSpec for the RAG tools
    """
    rag_tools = RAGTools(repo)
    
    # Create individual tool functions that can be registered
    def index_repository_tool(force: bool = False) -> str:
        return rag_tools.index_repository(force=force)
    
    def search_code_tool(query: str, top_k: int = 5, hybrid_alpha: float = 0.5) -> str:
        return rag_tools.search_code(query=query, top_k=top_k, hybrid_alpha=hybrid_alpha)
    
    def get_chunk_details_tool(chunk_id: str) -> str:
        return rag_tools.get_chunk_details(chunk_id=chunk_id)
    
    # Return the tool specification with proper function references
    return ToolSpec(
        function=search_code_tool,
        permission="read"
    )