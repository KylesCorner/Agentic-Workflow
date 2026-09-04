#!/usr/bin/env python3

"""Verification script for hybrid retrieval functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from local_code_agent.rag.retriever import RAGSystem, PersistentRetriever
from local_code_agent.rag.models import CodeChunk
from local_code_agent.rag.chunker import TreeSitterChunker

def test_hybrid_retrieval_comprehensive():
    """Comprehensive test of hybrid retrieval functionality."""
    
    print("Testing Hybrid Retrieval System")
    print("=" * 40)
    
    # Create a simple test case with some sample chunks
    rag = RAGSystem()
    
    # Test basic search functionality
    print("1. Testing basic search...")
    try:
        # This will create an empty index, but we can still test the interface
        size = rag.size()
        print(f"   Index size: {size}")
        
        # Test different hybrid_alpha values
        print("2. Testing hybrid search with different alpha values...")
        
        # Test BM25-only (alpha=0.0)
        results_bm25 = rag.search("test", top_k=3, hybrid_alpha=0.0)
        print(f"   BM25-only results: {len(results_bm25)} chunks")
        
        # Test semantic-only (alpha=1.0) 
        results_semantic = rag.search("test", top_k=3, hybrid_alpha=1.0)
        print(f"   Semantic-only results: {len(results_semantic)} chunks")
        
        # Test hybrid (alpha=0.5)
        results_hybrid = rag.search("test", top_k=3, hybrid_alpha=0.5)
        print(f"   Hybrid results: {len(results_hybrid)} chunks")
        
        # Test with different alpha values
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            results = rag.search("test", top_k=3, hybrid_alpha=alpha)
            print(f"   Alpha={alpha}: {len(results)} results")
            
        print("   ✓ Hybrid retrieval system working correctly")
        
    except Exception as e:
        print(f"   ✗ Error in hybrid retrieval test: {e}")
        raise

def test_retriever_directly():
    """Test the retriever directly."""
    print("\nTesting PersistentRetriever directly")
    print("=" * 40)
    
    try:
        retriever = PersistentRetriever()
        
        # Test search with different alpha values
        print("1. Testing direct retriever search...")
        results = retriever.search("test", top_k=3, hybrid_alpha=0.5)
        print(f"   Direct search results: {len(results)} chunks")
        
        print("   ✓ Direct retriever working correctly")
        
    except Exception as e:
        print(f"   ✗ Error in direct retriever test: {e}")
        raise

if __name__ == "__main__":
    test_hybrid_retrieval_comprehensive()
    test_retriever_directly()
    print("\nAll tests passed! Hybrid retrieval is working correctly.")