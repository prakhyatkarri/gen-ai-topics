import logging
from langchain_core.documents import Document


class DocumentService(logging.Logger):

    def __init__(self, logger):
        self.logger = logger

    def create_document(self, document: str, document_id: str, metadata: dict):
        return Document(page_content=document, metadata=metadata, id=document_id)
