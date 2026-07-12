import logging
from pyspark.sql import SparkSession

class DeltaWriter():

    def __init__(self, logger: logging.Logger):
        self.logger = logger

        if logger:
            self.logger.info("Logger successfully initiated!")

    def write_to_delta_table(self, spark: SparkSession, contents: dict, table_name: str):
        from pyspark.sql import functions as F
        # from pyspark.sql.types import *
        import uuid
        from datetime import datetime

        # ------------------------------------------------------------------------------
        # Example input:
        # chunks = [
        #     {
        #         "document_id": "invoice_001",
        #         "file_name": "invoice_001.pdf",
        #         "file_path": "/Volumes/raw/invoice_001.pdf",
        #         "page_number": 1,
        #         "chunk_index": 0,
        #         "chunk_text": "...",
        #         "token_count": 235,
        #     },
        # ]
        # ------------------------------------------------------------------------------

        run_id = str(uuid.uuid4())
        ingestion_time = datetime.utcnow()

        df = (
            spark.createDataFrame(contents)
                # .withColumn("chunk_id", F.expr("uuid()"))
                .withColumn("run_id", F.lit(run_id))
                .withColumn("ingestion_timestamp", F.lit(ingestion_time))
                # .withColumn("chunk_length", F.length("chunk_text"))
                .withColumn("content", contents['content'])
        )

        # # Recommended column ordering
        # df = df.select(
        #     "chunk_id",                 # Unique chunk identifier
        #     "document_id",              # Stable document identifier
        #     "file_name",                # Original PDF name
        #     "file_path",                # Storage location
        #     "page_number",              # PDF page
        #     "chunk_index",              # Chunk order within document
        #     "chunk_text",               # Text used for RAG
        #     "token_count",              # Token count (if available)
        #     "chunk_length",             # Character count
        #     "run_id",                   # Ingestion run identifier
        #     "ingestion_timestamp"       # Audit timestamp
        # )

        # (
        #     df.write
        #     .format("delta")
        #     .mode("append")
        #     .option("mergeSchema", "true")
        #     .saveAsTable("catalog_name.schema_name.pdf_chunks")
        # )
        print(df)