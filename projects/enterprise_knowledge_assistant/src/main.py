import logging
import sys
from pathlib import Path
from datetime import datetime
from document_service import DocumentService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # Set threshold for this handler
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

document_service = DocumentService(logger)


files = [f for f in Path(DocumentService.DOCUMENTS_PATH).iterdir() if f.is_file()]
contents = []
for file in files:
    file_name = str(file)
    content = document_service.read_document(file_name)
    metadata = {
        "file_name": file_name,
        "size": len(content),
        "timestamp": str(datetime.now())
    }
    contents.append(
        {
            "content": content,
            "metadata": metadata
        }
    )

for c in contents: 
    print(c['metadata'])