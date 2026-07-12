import logging
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import uuid


class ChunkService:

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def chunk_documents_by_length(
        self,
        documents: list[Document],
        chunk_size: int = 10,
        chunk_overlap_size: int = 5,
    ) -> list[Document]:
        """Chunk list of documents by length
        Args:
            documents: list of Documents to be chunked
        Returns:
            chunked_documents: list of chunked documents
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap_size,
            length_function=len,
            is_separator_regex=False,
        )

        metadatas = []

        for i, _ in enumerate(documents):
            metadatas.append({"document_id": f"doc_{i}_{uuid.uuid4()}"})

        return text_splitter.split_documents(
            # text_splitter.create_documents(documents, metadatas=metadatas)
            documents
        )

    def chunk_text_by_length(
        self,
        documents: list[str],
        chunk_size: int = 10,
        chunk_overlap_size: int = 5,
    ) -> list[str]:
        """Chunk list of strings by length
        Args:
            documents: list of strings to be chunked
        Returns:
            chunked_strings: list of chunked strings
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap_size,
            length_function=len,
            is_separator_regex=False,
        )

        chunked_documents = []
        for index, item in enumerate(documents):
            metadata = item['metadata']
            metadata['chunk_id']= index
            chunked_documents.append({
                'content': text_splitter.split_text(item['content']),
                'metadata': metadata
            })

        return chunked_documents
