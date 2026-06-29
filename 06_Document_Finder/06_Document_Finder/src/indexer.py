"""Document Indexer Module

Creates and manages search indices for document content.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Set, Optional
from collections import defaultdict
import re
from pathlib import Path


class DocumentIndexer:
    """Creates and manages inverted index for document search."""
    
    def __init__(self, index_dir: str = "./index", logger: Optional[logging.Logger] = None):
        """Initialize DocumentIndexer.
        
        Args:
            index_dir: Directory to store index files
            logger: Optional logger instance
        """
        self.index_dir = index_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Inverted index: {term: {doc_id: [positions]}}
        self.inverted_index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        # Document metadata: {doc_id: metadata}
        self.documents: Dict[str, Dict] = {}
        
        # Document count
        self.doc_count = 0
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable terms.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens and common stop words
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
                     'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
                     'that', 'the', 'to', 'was', 'will', 'with'}
        
        tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
        
        return tokens
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional document metadata
        """
        if doc_id in self.documents:
            self.logger.warning(f"Document {doc_id} already exists. Updating...")
            self.remove_document(doc_id)
        
        # Tokenize content
        tokens = self._tokenize(content)
        
        # Build inverted index with term positions
        for position, token in enumerate(tokens):
            self.inverted_index[token][doc_id].append(position)
        
        # Store document metadata
        self.documents[doc_id] = {
            'doc_id': doc_id,
            'content': content,
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            **(metadata or {})
        }
        
        self.doc_count += 1
        self.logger.info(f"Indexed document: {doc_id}")
    
    def remove_document(self, doc_id: str):
        """Remove a document from the index.
        
        Args:
            doc_id: Document identifier to remove
        """
        if doc_id not in self.documents:
            self.logger.warning(f"Document {doc_id} not found in index")
            return
        
        # Remove from inverted index
        for term in list(self.inverted_index.keys()):
            if doc_id in self.inverted_index[term]:
                del self.inverted_index[term][doc_id]
                # Remove term if no documents left
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
        
        # Remove document metadata
        del self.documents[doc_id]
        self.doc_count -= 1
        
        self.logger.info(f"Removed document: {doc_id}")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of matching documents with scores
        """
        if not self.documents:
            self.logger.warning("Index is empty")
            return []
        
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            self.logger.warning("Query produced no valid tokens")
            return []
        
        # Calculate TF-IDF scores
        doc_scores = defaultdict(float)
        
        for token in query_tokens:
            if token in self.inverted_index:
                # IDF: log(total_docs / docs_containing_term)
                idf = self.doc_count / len(self.inverted_index[token])
                
                for doc_id, positions in self.inverted_index[token].items():
                    # TF: term frequency in document
                    tf = len(positions) / self.documents[doc_id]['token_count']
                    
                    # TF-IDF score
                    doc_scores[doc_id] += tf * idf
        
        # Sort by score
        ranked_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Prepare results
        results = []
        for doc_id, score in ranked_docs:
            doc = self.documents[doc_id].copy()
            doc['score'] = score
            doc['matched_terms'] = [
                token for token in query_tokens 
                if token in self.inverted_index and doc_id in self.inverted_index[token]
            ]
            results.append(doc)
        
        return results
    
    def search_phrase(self, phrase: str, top_k: int = 10) -> List[Dict]:
        """Search for exact phrase matches.
        
        Args:
            phrase: Exact phrase to search for
            top_k: Number of top results to return
            
        Returns:
            List of documents containing the phrase
        """
        phrase_tokens = self._tokenize(phrase)
        
        if not phrase_tokens:
            return []
        
        # Find documents containing all tokens
        first_token = phrase_tokens[0]
        if first_token not in self.inverted_index:
            return []
        
        candidate_docs = set(self.inverted_index[first_token].keys())
        
        # Filter to documents containing all tokens
        for token in phrase_tokens[1:]:
            if token in self.inverted_index:
                candidate_docs &= set(self.inverted_index[token].keys())
            else:
                return []
        
        # Check for consecutive positions
        matching_docs = []
        
        for doc_id in candidate_docs:
            # Get positions of first token
            first_positions = self.inverted_index[first_token][doc_id]
            
            # Check if all tokens appear consecutively
            for start_pos in first_positions:
                match = True
                for i, token in enumerate(phrase_tokens[1:], 1):
                    expected_pos = start_pos + i
                    if expected_pos not in self.inverted_index[token][doc_id]:
                        match = False
                        break
                
                if match:
                    doc = self.documents[doc_id].copy()
                    doc['score'] = 1.0  # Exact match
                    doc['matched_phrase'] = phrase
                    matching_docs.append(doc)
                    break
        
        return matching_docs[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        return self.documents.get(doc_id)
    
    def get_stats(self) -> Dict:
        """Get index statistics.
        
        Returns:
            Dictionary of index statistics
        """
        return {
            'document_count': self.doc_count,
            'unique_terms': len(self.inverted_index),
            'index_size_mb': self._estimate_size() / (1024 * 1024),
            'avg_doc_length': sum(d['token_count'] for d in self.documents.values()) / max(self.doc_count, 1)
        }
    
    def _estimate_size(self) -> int:
        """Estimate index size in bytes."""
        # Rough estimation
        size = 0
        for term, docs in self.inverted_index.items():
            size += len(term) * 2  # Unicode characters
            for doc_id, positions in docs.items():
                size += len(doc_id) * 2
                size += len(positions) * 8  # Integer positions
        return size
    
    def save_index(self, name: str = "index"):
        """Save index to disk.
        
        Args:
            name: Base name for index files
        """
        index_path = os.path.join(self.index_dir, f"{name}.pkl")
        docs_path = os.path.join(self.index_dir, f"{name}_docs.pkl")
        
        try:
            # Save inverted index
            with open(index_path, 'wb') as f:
                pickle.dump(dict(self.inverted_index), f)
            
            # Save documents
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            self.logger.info(f"Index saved to {self.index_dir}")
        
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, name: str = "index"):
        """Load index from disk.
        
        Args:
            name: Base name for index files
        """
        index_path = os.path.join(self.index_dir, f"{name}.pkl")
        docs_path = os.path.join(self.index_dir, f"{name}_docs.pkl")
        
        try:
            # Load inverted index
            with open(index_path, 'rb') as f:
                loaded_index = pickle.load(f)
                self.inverted_index = defaultdict(lambda: defaultdict(list), loaded_index)
            
            # Load documents
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            self.doc_count = len(self.documents)
            self.logger.info(f"Index loaded from {self.index_dir}")
        
        except FileNotFoundError:
            self.logger.warning(f"Index files not found in {self.index_dir}")
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            raise
    
    def clear_index(self):
        """Clear the entire index."""
        self.inverted_index.clear()
        self.documents.clear()
        self.doc_count = 0
        self.logger.info("Index cleared")