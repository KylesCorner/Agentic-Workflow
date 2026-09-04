#!/usr/bin/env python3
"""
Integration test for the Tree-sitter RAG system.

This test verifies that all components (chunker, document store, embeddings, retriever)
work together properly in a complete end-to-end workflow.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from local_code_agent.rag.chunker import TreeSitterChunker
from local_code_agent.rag.retriever import RAGSystem, PersistentRetriever
from local_code_agent.rag.models import CodeChunk
from local_code_agent.rag.document_store import SimpleDocumentStore
from local_code_agent.rag.embeddings import EmbeddingSystem

def create_test_files(temp_dir):
    """Create some test code files for indexing."""
    
    # Create a Python file with some functions
    python_content = '''def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True

class Calculator:
    """A calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b

def complex_function(x, y, z=None):
    """A complex function with multiple parameters."""
    if z is None:
        z = 0
    result = x + y + z
    return result
'''
    
    # Create a C file with some functions
    c_content = '''#include <stdio.h>
#include <stdlib.h>

int add_numbers(int a, int b) {
    return a + b;
}

void print_message(const char* message) {
    printf("Message: %s\\n", message);
}

int main() {
    int sum = add_numbers(5, 3);
    printf("Sum: %d\\n", sum);
    return 0;
}
'''
    
    # Create test files
    py_file = temp_dir / "test_module.py"
    c_file = temp_dir / "test_module.c"
    
    py_file.write_text(python_content)
    c_file.write_text(c_content)
    
    return [py_file, c_file]

def test_chunker_integration():
    """Test that the chunker works correctly."""
    print("Testing Tree-sitter Chunker integration...")
    
    chunker = TreeSitterChunker()
    
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = create_test_files(temp_path)
        
        chunks = []
        for file_path in test_files:
            try:
                file_chunks = chunker.chunk_file(file_path, temp_path)
                chunks.extend(file_chunks)
                print(f"  Chunked {file_path.name}: {len(file_chunks)} chunks")
            except Exception as e:
                print(f"  Error chunking {file_path.name}: {e}")
                return False
        
        # Verify we got some chunks
        if len(chunks) == 0:
            print("  ERROR: No chunks generated!")
            return False
            
        print(f"  ✓ Successfully chunked {len(chunks)} total chunks")
        
        # Check that chunks have required fields
        for i, chunk in enumerate(chunks[:3]):  # Check first 3 chunks
            if not all([chunk.id, chunk.path, chunk.language, chunk.content]):
                print(f"  ERROR: Chunk {i} missing required fields")
                return False
        
        print("  ✓ All chunks have required fields")
        return True

def test_document_store_integration():
    """Test that the document store works correctly."""
    print("Testing Document Store integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        
        # Create some test chunks
        chunks = [
            CodeChunk(
                id="chunk1",
                path="test.py",
                language="python",
                start_line=1,
                end_line=5,
                start_byte=0,
                end_byte=100,
                node_type="function_definition",
                symbol="hello_world",
                content='def hello_world():\n    """A simple function."""\n    print("Hello, World!")\n    return True',
                parent_symbol=None
            ),
            CodeChunk(
                id="chunk2",
                path="test.py", 
                language="python",
                start_line=7,
                end_line=12,
                start_byte=100,
                end_byte=200,
                node_type="class_definition",
                symbol="Calculator",
                content='class Calculator:\n    """A calculator class."""\n    \n    def add(self, a, b):\n        """Add two numbers."""\n        return a + b',
                parent_symbol=None
            )
        ]
        
        # Test adding chunks
        from local_code_agent.rag.document_store import SimpleDocumentStore
        store = SimpleDocumentStore(db_path)
        
        try:
            store.add_chunks(chunks)
            
            # Test retrieving chunks
            all_chunks = store.get_all_chunks()
            if len(all_chunks) != 2:
                print(f"  ERROR: Expected 2 chunks, got {len(all_chunks)}")
                return False
                
            # Test getting specific chunk
            retrieved_chunk = store.get_chunk("chunk1")
            if not retrieved_chunk or retrieved_chunk.id != "chunk1":
                print("  ERROR: Could not retrieve specific chunk")
                return False
            
            # Test size
            size = store.size()
            if size != 2:
                print(f"  ERROR: Expected size 2, got {size}")
                return False
                
            print("  ✓ Document store operations working correctly")
            return True
            
        except Exception as e:
            print(f"  ERROR: Document store test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_embedding_system_integration():
    """Test that the embedding system works correctly."""
    print("Testing Embedding System integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        
        # Create some test chunks
        chunks = [
            CodeChunk(
                id="chunk1",
                path="test.py",
                language="python",
                start_line=1,
                end_line=5,
                start_byte=0,
                end_byte=100,
                node_type="function_definition",
                symbol="hello_world",
                content='def hello_world():\n    """A simple function."""\n    print("Hello, World!")\n    return True',
                parent_symbol=None
            )
        ]
        
        try:
            from local_code_agent.rag.embeddings import EmbeddingSystem
            
            # Test embedding system
            embedding_system = EmbeddingSystem(db_path)
            
            # Add chunks with embeddings
            embedding_system.add_chunks_with_embeddings(chunks)
            
            # Check that we can retrieve the embedding
            embedding = embedding_system.embedding_store.get_embedding("chunk1")
            if not embedding:
                print("  ERROR: Could not retrieve embedding")
                return False
                
            # Check embedding dimensions (should be a list of floats)
            if not isinstance(embedding, list) or len(embedding) == 0:
                print("  ERROR: Invalid embedding format")
                return False
                
            print(f"  ✓ Embedding system working correctly (vector size: {len(embedding)})")
            return True
            
        except Exception as e:
            print(f"  ERROR: Embedding system test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_retriever_integration():
    """Test that the retriever works correctly."""
    print("Testing Retriever integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        
        # Create some test chunks
        chunks = [
            CodeChunk(
                id="chunk1",
                path="test.py",
                language="python",
                start_line=1,
                end_line=5,
                start_byte=0,
                end_byte=100,
                node_type="function_definition",
                symbol="hello_world",
                content='def hello_world():\n    """A simple function."""\n    print("Hello, World!")\n    return True',
                parent_symbol=None
            ),
            CodeChunk(
                id="chunk2",
                path="test.py", 
                language="python",
                start_line=7,
                end_line=12,
                start_byte=100,
                end_byte=200,
                node_type="class_definition",
                symbol="Calculator",
                content='class Calculator:\n    """A calculator class."""\n    \n    def add(self, a, b):\n        """Add two numbers."""\n        return a + b',
                parent_symbol=None
            )
        ]
        
        try:
            from local_code_agent.rag.retriever import PersistentRetriever
            
            # Test retriever
            retriever = PersistentRetriever(db_path)
            
            # Add chunks with embeddings
            retriever.add_chunks_with_embeddings(chunks)
            
            # Test search functionality
            results = retriever.search("function", top_k=5)
            print(f"  ✓ Search returned {len(results)} results")
            
            # Test different alpha values
            bm25_results = retriever.search("function", top_k=3, hybrid_alpha=0.0)
            semantic_results = retriever.search("function", top_k=3, hybrid_alpha=1.0)
            hybrid_results = retriever.search("function", top_k=3, hybrid_alpha=0.5)
            
            print(f"  ✓ BM25-only search: {len(bm25_results)} results")
            print(f"  ✓ Semantic-only search: {len(semantic_results)} results")  
            print(f"  ✓ Hybrid search: {len(hybrid_results)} results")
            
            # Verify results have proper structure
            if results:
                result = results[0]
                if not hasattr(result, 'chunk') or not hasattr(result, 'score'):
                    print("  ERROR: Invalid result structure")
                    return False
                    
            print("  ✓ Retriever working correctly")
            return True
            
        except Exception as e:
            print(f"  ERROR: Retriever test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_full_rag_system():
    """Test the complete RAG system integration."""
    print("Testing Full RAG System integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create some test files
            test_files = create_test_files(Path(temp_dir))
            
            # Test full RAG system
            rag_system = RAGSystem(db_path=os.path.join(temp_dir, "rag_index.db"))
            
            # Index the repository (this tests chunker + retriever integration)
            rag_system.index_repository(Path(temp_dir))
            
            # Test search functionality
            results = rag_system.search("function", top_k=5)
            print(f"  ✓ Full system search returned {len(results)} results")
            
            # Test different search modes
            bm25_results = rag_system.search("function", top_k=3, hybrid_alpha=0.0)
            semantic_results = rag_system.search("function", top_k=3, hybrid_alpha=1.0)
            hybrid_results = rag_system.search("function", top_k=3, hybrid_alpha=0.5)
            
            print(f"  ✓ BM25-only search: {len(bm25_results)} results")
            print(f"  ✓ Semantic-only search: {len(semantic_results)} results")  
            print(f"  ✓ Hybrid search: {len(hybrid_results)} results")
            
            # Test getting specific chunks
            all_chunks = rag_system.get_all_chunks()
            print(f"  ✓ Retrieved {len(all_chunks)} total chunks")
            
            size = rag_system.size()
            print(f"  ✓ System size: {size} chunks")
            
            print("  ✓ Full RAG system working correctly")
            return True
            
        except Exception as e:
            print(f"  ERROR: Full RAG system test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run all integration tests."""
    print("Running Tree-sitter RAG System Integration Tests")
    print("=" * 50)
    
    tests = [
        test_chunker_integration,
        test_document_store_integration,
        test_embedding_system_integration,
        test_retriever_integration,
        test_full_rag_system
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
    
    print("=" * 50)
    print(f"Integration Tests Results:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")
    
    if failed == 0:
        print("✓ All integration tests passed!")
        return 0
    else:
        print("✗ Some integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())