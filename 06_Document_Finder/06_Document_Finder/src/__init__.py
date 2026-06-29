"""Document Search Engine Package

A production-grade document search engine supporting PDF, Word, and Markdown files.
"""

__version__ = "1.0.0"
__author__ = "Databricks"

from .search_engine import DocumentSearchEngine
from .document_processor import DocumentProcessor
from .indexer import DocumentIndexer

__all__ = ["DocumentSearchEngine", "DocumentProcessor", "DocumentIndexer"]