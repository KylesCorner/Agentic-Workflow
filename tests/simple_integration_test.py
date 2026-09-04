#!/usr/bin/env python3
"""Simple integration test for the RAG system."""

import sys
import os
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that all modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from local_code_agent.rag.chunker import TreeSitterChunker
        from local_code_agent.rag.retriever import RAGSystem, PersistentRetriever
        from local_code_agent.rag.models import CodeChunk
        from local_code_agent.rag.document_store import SimpleDocumentStore
        from local_code_agent.rag.embeddings import EmbeddingSystem
        
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunker_basic():
    """Test basic chunker functionality."""
    print("Testing chunker basic functionality...")
    
    try:
        from local_code_agent.rag.chunker import TreeSitterChunker
        
        chunker = TreeSitterChunker()
        print("✓ Chunker created successfully")
        
        # Test that it has required methods
        methods = ['chunk_file', 'chunk_content']
        for method in methods:
            if not hasattr(chunker, method):
                print(f"✗ Missing method: {method}")
                return False
                
        print("✓ Chunker has required methods")
        return True
    except Exception as e:
        print(f"✗ Chunker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run basic integration tests."""
    print("Running Basic RAG System Integration Tests")
    print("=" * 45)
    
    tests = [
        test_basic_imports,
        test_chunker_basic
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"Running {test.__name__}...")
            if test():
                passed += 1
                print("  ✓ PASSED")
            else:
                failed += 1
                print("  ✗ FAILED")
        except Exception as e:
            print(f"  ERROR: Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 45)
    print(f"Basic Tests Results:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")
    
    if failed == 0:
        print("✓ All basic tests passed!")
        return 0
    else:
        print("✗ Some basic tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())