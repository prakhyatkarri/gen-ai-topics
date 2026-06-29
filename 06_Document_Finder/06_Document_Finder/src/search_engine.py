"""Document Search Engine Module

Main interface for the document search engine.
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

from .document_processor import DocumentProcessor
from .indexer import DocumentIndexer


class DocumentSearchEngine:
    """Production-grade document search engine.
    
    Supports searching through PDF, Word, and Markdown documents.
    """
    
    def __init__(self, index_dir: str = "./index", log_level: str = "INFO"):
        """Initialize Document Search Engine.
        
        Args:
            index_dir: Directory to store search index
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Initialize components
        self.processor = DocumentProcessor(logger=self.logger)
        self.indexer = DocumentIndexer(index_dir=index_dir, logger=self.logger)
        
        self.logger.info("Document Search Engine initialized")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger for the search engine."""
        logger = logging.getLogger('DocumentSearchEngine')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def index_document(self, file_path: str) -> bool:
        """Index a single document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process document
            doc_data = self.processor.process_document(file_path)
            
            # Add to index
            doc_id = os.path.abspath(file_path)
            metadata = {
                'file_name': doc_data['file_name'],
                'file_type': doc_data['file_type'],
                'size': doc_data['size'],
                'word_count': doc_data['word_count']
            }
            
            self.indexer.add_document(
                doc_id=doc_id,
                content=doc_data['content'],
                metadata=metadata
            )
            
            self.logger.info(f"Successfully indexed: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to index {file_path}: {str(e)}")
            return False
    
    def index_directory(self, directory: str, recursive: bool = True, save: bool = True) -> Dict:
        """Index all documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            save: Whether to save index after indexing
            
        Returns:
            Dictionary with indexing statistics
        """
        self.logger.info(f"Indexing directory: {directory}")
        
        # Process all documents
        documents = self.processor.process_directory(directory, recursive=recursive)
        
        # Add to index
        success_count = 0
        fail_count = 0
        
        for doc_data in documents:
            try:
                doc_id = os.path.abspath(doc_data['file_path'])
                metadata = {
                    'file_name': doc_data['file_name'],
                    'file_type': doc_data['file_type'],
                    'size': doc_data['size'],
                    'word_count': doc_data['word_count']
                }
                
                self.indexer.add_document(
                    doc_id=doc_id,
                    content=doc_data['content'],
                    metadata=metadata
                )
                success_count += 1
            
            except Exception as e:
                self.logger.error(f"Failed to index {doc_data['file_path']}: {str(e)}")
                fail_count += 1
        
        # Save index
        if save and success_count > 0:
            self.indexer.save_index()
        
        stats = {
            'total_processed': len(documents),
            'successful': success_count,
            'failed': fail_count,
            'index_stats': self.indexer.get_stats()
        }
        
        self.logger.info(f"Indexing complete: {success_count} successful, {fail_count} failed")
        return stats
    
    def search(self, query: str, top_k: int = 10, phrase_search: bool = False) -> List[Dict]:
        """Search for documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            phrase_search: If True, search for exact phrase
            
        Returns:
            List of matching documents with scores
        """
        self.logger.info(f"Searching for: '{query}'")
        
        if phrase_search:
            results = self.indexer.search_phrase(query, top_k=top_k)
        else:
            results = self.indexer.search(query, top_k=top_k)
        
        self.logger.info(f"Found {len(results)} matching documents")
        return results
    
    def get_document_preview(self, doc_id: str, max_length: int = 500) -> Optional[str]:
        """Get a preview of document content.
        
        Args:
            doc_id: Document identifier
            max_length: Maximum preview length in characters
            
        Returns:
            Document preview or None if not found
        """
        doc = self.indexer.get_document(doc_id)
        if doc and 'content' in doc:
            content = doc['content']
            if len(content) > max_length:
                return content[:max_length] + "..."
            return content
        return None
    
    def get_document_snippet(self, doc_id: str, query: str, context_length: int = 100) -> Optional[str]:
        """Get a snippet of document content around query match.
        
        Args:
            doc_id: Document identifier
            query: Search query
            context_length: Characters of context before/after match
            
        Returns:
            Document snippet or None if not found
        """
        doc = self.indexer.get_document(doc_id)
        if not doc or 'content' not in doc:
            return None
        
        content = doc['content'].lower()
        query_lower = query.lower()
        
        # Find first occurrence
        pos = content.find(query_lower)
        if pos == -1:
            # Fallback to first word
            words = query_lower.split()
            for word in words:
                pos = content.find(word)
                if pos != -1:
                    break
        
        if pos == -1:
            # Return beginning if no match found
            return doc['content'][:context_length * 2] + "..."
        
        # Extract snippet with context
        start = max(0, pos - context_length)
        end = min(len(doc['content']), pos + len(query) + context_length)
        
        snippet = doc['content'][start:end]
        
        if start > 0:
            snippet = "..." + snippet
        if end < len(doc['content']):
            snippet = snippet + "..."
        
        return snippet
    
    def remove_document(self, file_path: str) -> bool:
        """Remove a document from the index.
        
        Args:
            file_path: Path to the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = os.path.abspath(file_path)
            self.indexer.remove_document(doc_id)
            self.logger.info(f"Removed document: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove {file_path}: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get search engine statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.indexer.get_stats()
    
    def save_index(self, name: str = "index"):
        """Save the search index to disk.
        
        Args:
            name: Name for the index files
        """
        self.indexer.save_index(name)
        self.logger.info(f"Index saved as '{name}'")
    
    def load_index(self, name: str = "index"):
        """Load a search index from disk.
        
        Args:
            name: Name of the index files
        """
        self.indexer.load_index(name)
        self.logger.info(f"Index '{name}' loaded")
    
    def clear_index(self):
        """Clear all documents from the index."""
        self.indexer.clear_index()
        self.logger.info("Index cleared")
    
    def list_indexed_documents(self) -> List[Dict]:
        """Get list of all indexed documents.
        
        Returns:
            List of document metadata
        """
        documents = []
        for doc_id, doc_data in self.indexer.documents.items():
            documents.append({
                'doc_id': doc_id,
                'file_name': doc_data.get('file_name', 'Unknown'),
                'file_type': doc_data.get('file_type', 'Unknown'),
                'word_count': doc_data.get('word_count', 0),
                'size': doc_data.get('size', 0)
            })
        return documents