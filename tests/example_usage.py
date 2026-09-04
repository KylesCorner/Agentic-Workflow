#!/usr/bin/env python3

"""Example usage of the Tree-sitter RAG system."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from local_code_agent.rag.retriever import RAGSystem

def main():
    # Create a simple example to demonstrate usage
    print("Tree-sitter RAG System Example")
    print("=" * 40)
    
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Show what's available
    print("Available modules:")
    print("- TreeSitterChunker: For extracting semantic code chunks")
    print("- RAGSystem: Main interface for indexing and searching")
    
    print("\nUsage example:")
    print("1. Create RAG system: rag = RAGSystem()")
    print("2. Index repository: rag.index_repository('/path/to/repo')")
    print("3. Search code with hybrid retrieval:")
    print("   # BM25-only search")
    print("   results = rag.search('function name', top_k=5, hybrid_alpha=0.0)")
    print("   # Semantic-only search")  
    print("   results = rag.search('function name', top_k=5, hybrid_alpha=1.0)")
    print("   # Hybrid search (default)")
    print("   results = rag.search('function name', top_k=5, hybrid_alpha=0.5)")
    
    # Show the structure
    print(f"\nRAG System Components:")
    print(f"- Chunker: {type(rag.chunker).__name__}")
    print(f"- Retriever: {type(rag.retriever).__name__}")
    
    print("\nHybrid Retrieval Features:")
    print("- BM25-based text matching for lexical similarity")
    print("- Semantic similarity using embeddings")
    print("- Weighted combination of both approaches")
    print("- Configurable alpha parameter (0.0 to 1.0)")
    
    print("\nTool Integration:")
    print("The RAG system is now integrated into the toolset as:")
    print("- index_repository: Index repository for semantic search")
    print("- search_code: Search code with hybrid retrieval")
    print("- get_chunk_details: Get detailed information about specific chunks")

if __name__ == "__main__":
    main()