"""Unit tests for DocumentSearchEngine."""

import pytest
import os
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search_engine import DocumentSearchEngine


class TestDocumentSearchEngine:
    """Test suite for DocumentSearchEngine."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def engine(self, temp_dir):
        """Create search engine instance."""
        return DocumentSearchEngine(index_dir=temp_dir, log_level="ERROR")
    
    @pytest.fixture
    def sample_docs(self, temp_dir):
        """Create sample documents for testing."""
        docs_dir = os.path.join(temp_dir, "docs")
        os.makedirs(docs_dir)
        
        # Create test documents
        docs = [
            ("doc1.md", "# Machine Learning\n\nMachine learning is great."),
            ("doc2.md", "# Python\n\nPython is a programming language."),
            ("doc3.md", "# Data Science\n\nData science uses machine learning.")
        ]
        
        for filename, content in docs:
            with open(os.path.join(docs_dir, filename), 'w') as f:
                f.write(content)
        
        return docs_dir
    
    def test_initialization(self, temp_dir):
        """Test engine initialization."""
        engine = DocumentSearchEngine(index_dir=temp_dir)
        assert engine is not None
        assert engine.indexer is not None
        assert engine.processor is not None
    
    def test_index_document(self, engine, temp_dir):
        """Test indexing a single document."""
        # Create test file
        test_file = os.path.join(temp_dir, "test.md")
        with open(test_file, 'w') as f:
            f.write("# Test\n\nTest content")
        
        # Index it
        success = engine.index_document(test_file)
        assert success is True
        
        # Verify it's indexed
        stats = engine.get_statistics()
        assert stats['document_count'] == 1
    
    def test_index_directory(self, engine, sample_docs):
        """Test indexing a directory."""
        stats = engine.index_directory(sample_docs, recursive=False)
        
        assert stats['successful'] == 3
        assert stats['failed'] == 0
        assert stats['total_processed'] == 3
    
    def test_search(self, engine, sample_docs):
        """Test basic search functionality."""
        # Index documents
        engine.index_directory(sample_docs)
        
        # Search
        results = engine.search("machine learning", top_k=10)
        
        assert len(results) > 0
        assert all('score' in r for r in results)
        assert all('file_name' in r for r in results)
    
    def test_search_empty_index(self, engine):
        """Test search on empty index."""
        results = engine.search("test query")
        assert len(results) == 0
    
    def test_phrase_search(self, engine, sample_docs):
        """Test phrase search."""
        engine.index_directory(sample_docs)
        
        # Phrase that exists
        results = engine.search("machine learning", phrase_search=True)
        assert len(results) >= 1
        
        # Phrase that doesn't exist
        results = engine.search("nonexistent phrase", phrase_search=True)
        assert len(results) == 0
    
    def test_remove_document(self, engine, temp_dir):
        """Test removing a document."""
        # Create and index document
        test_file = os.path.join(temp_dir, "test.md")
        with open(test_file, 'w') as f:
            f.write("# Test\n\nContent")
        
        engine.index_document(test_file)
        assert engine.get_statistics()['document_count'] == 1
        
        # Remove it
        success = engine.remove_document(test_file)
        assert success is True
        assert engine.get_statistics()['document_count'] == 0
    
    def test_save_and_load_index(self, engine, sample_docs):
        """Test index persistence."""
        # Index documents
        engine.index_directory(sample_docs)
        original_count = engine.get_statistics()['document_count']
        
        # Save
        engine.save_index("test_index")
        
        # Clear and reload
        engine.clear_index()
        assert engine.get_statistics()['document_count'] == 0
        
        engine.load_index("test_index")
        assert engine.get_statistics()['document_count'] == original_count
    
    def test_list_documents(self, engine, sample_docs):
        """Test listing indexed documents."""
        engine.index_directory(sample_docs)
        
        docs = engine.list_indexed_documents()
        assert len(docs) == 3
        assert all('file_name' in d for d in docs)
        assert all('file_type' in d for d in docs)
    
    def test_get_document_preview(self, engine, temp_dir):
        """Test document preview."""
        test_file = os.path.join(temp_dir, "test.md")
        content = "This is a long test content. " * 100
        with open(test_file, 'w') as f:
            f.write(content)
        
        engine.index_document(test_file)
        doc_id = os.path.abspath(test_file)
        
        preview = engine.get_document_preview(doc_id, max_length=50)
        assert len(preview) <= 53  # 50 + "..."
    
    def test_get_document_snippet(self, engine, temp_dir):
        """Test document snippet extraction."""
        test_file = os.path.join(temp_dir, "test.md")
        with open(test_file, 'w') as f:
            f.write("Some text before. Machine learning is important. Some text after.")
        
        engine.index_document(test_file)
        doc_id = os.path.abspath(test_file)
        
        snippet = engine.get_document_snippet(doc_id, "machine learning", context_length=20)
        assert "machine learning" in snippet.lower()
    
    def test_statistics(self, engine, sample_docs):
        """Test statistics retrieval."""
        engine.index_directory(sample_docs)
        
        stats = engine.get_statistics()
        assert 'document_count' in stats
        assert 'unique_terms' in stats
        assert 'index_size_mb' in stats
        assert stats['document_count'] == 3
        assert stats['unique_terms'] > 0
    
    def test_clear_index(self, engine, sample_docs):
        """Test clearing the index."""
        engine.index_directory(sample_docs)
        assert engine.get_statistics()['document_count'] > 0
        
        engine.clear_index()
        assert engine.get_statistics()['document_count'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])