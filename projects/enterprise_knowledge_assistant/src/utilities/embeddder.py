import os
import logging
from langchain_core.documents import Document


class EmbeddingService(logging.Logger):
    def __init__(self, logger):
        self.logger = logger

    def create_embedding(self, document: str):
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        model_name = "databricks-bge-large-en"

        print(f"Sending {len(document)} chunks to Databricks Model Serving...")
        response = w.serving_endpoints.query(name=model_name, input=document)

        response_dict = response.as_dict()
        embeddings = []
        for item in response_dict["data"]:
            embeddings.append(item['embedding'])
        return embeddings
