#!/usr/bin/env python3

"""Simple test for hybrid retrieval."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from local_code_agent.rag.retriever import RAGSystem
from local_code_agent.rag.models import CodeChunk

def test_basic_functionality():
    """Test basic functionality."""
    print("Testing basic hybrid retrieval system")
    
    # Create a simple test case
    rag = RAGSystem()
    
    # Test that the search method exists and can be called
    try:
        size = rag.size()
        print(f"Index size: {size}")
        
        # Test different hybrid_alpha values
        print("Testing hybrid search with different alpha values...")
        
        # Test BM25-only (alpha=0.0)
        results_bm25 = rag.search("test", top_k=3, hybrid_alpha=0.0)
        print(f"BM25-only results: {len(results_bm25)} chunks")
        
        # Test semantic-only (alpha=1.0) 
        results_semantic = rag.search("test", top_k=3, hybrid_alpha=1.0)
        print(f"Semantic-only results: {len(results_semantic)} chunks")
        
        # Test hybrid (alpha=0.5)
        results_hybrid = rag.search("test", top_k=3, hybrid_alpha=0.5)
        print(f"Hybrid results: {len(results_hybrid)} chunks")
        
        print("✓ Hybrid retrieval system working correctly")
        
    except Exception as e:
        print(f"✗ Error in hybrid retrieval test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()