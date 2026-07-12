import logging
import sys
from pathlib import Path
from datetime import datetime
# from writer.delta_writer import DeltaWriter
from reader.pdf_reader import PDFReader
from utilities.chunker import ChunkService
from utilities.documenter import DocumentService
from utilities.embeddder import EmbeddingService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # Set threshold for this handler
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)


# from pyspark.sql import SparkSession

# Initialize the SparkSession
# spark = SparkSession.builder \
#     .appName("My Spark Application") \
#     .master("local[*]") \
#     .getOrCreate()

# if spark:
#     logger.info("Spark successfully initialized !!!")


pdf_reader = PDFReader(logger)
datasets_path = "k:/Learning/Git/gen-ai-topics/projects/enterprise_knowledge_assistant/datasets"
files = [f for f in Path(datasets_path).iterdir() if f.is_file()]
contents = []
for id, file in enumerate(files, start=1):
    file_name = str(file)
    pdf_document = pdf_reader.read_document(file_name)
    content = pdf_document['content']
    metadata = {
        "document_id": f"document_datasets_{id}",
        "file_name": file_name,
        "file_path":  str(file.absolute()),
        "size": len(content),
        "timestamp": str(datetime.now())
    }
    
    contents.append(
        {
            "content": content,
            "metadata": metadata
        }
    )

document_service = DocumentService(logger)
chunkService = ChunkService(logger)

documents = []
for c in contents:     
    content = str(c['content']),
    metadata = c['metadata']
    document_id = metadata['document_id']
    for index, item in enumerate(content):
        documents.append(document_service.create_document(item, document_id, metadata))

texts = []
for c in contents:     
    content = str(c['content']),
    metadata = c['metadata']
    document_id = metadata['document_id']
    for index, item in enumerate(content):
        texts.append(
            {
                'content': item,
                "metadata": metadata
            }
        )


# # chunks = [
#         #     {
#         #         "document_id": "invoice_001",
#         #         "file_name": "invoice_001.pdf",
#         #         "file_path": "/Volumes/raw/invoice_001.pdf",
#         #         "page_number": 1,
#         #         "chunk_index": 0,
#         #         "chunk_text": "...",
#         #         "token_count": 235,
#         #     },
#         # ]    

# # delta_writer = DeltaWriter(logger)
# # delta_writer.write_to_delta_table(spark, contents, "")

# chunked_documents = chunkService.chunk_documents_by_length(documents, 10, 2)
chunked_documents = chunkService.chunk_text_by_length(texts)
# print(chunked_documents[0])

embedder_service = EmbeddingService(logger)

embeddings = []

# for chunk in chunked_documents:
chunk = chunked_documents[0]
embeddings.append({
    'embeddings': embedder_service.create_embedding(chunk['content']),
    'metadata': chunk['metadata']
})

print(len(embeddings))
