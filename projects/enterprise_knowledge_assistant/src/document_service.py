import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


try:
    import PyPDF2
except Exception as e:
    print("Cannot import PyPDF2: ", e)


class DocumentService:
    DOCUMENTS_PATH = "datasets/"

    def __init__(self, logger: Optional[logging.Logger]):
        self.logger = logger
        self.__validate_dependencies(logger)

    def __validate_dependencies(self, logger: Optional[logging.Logger]):
        if PyPDF2 is None:
            self.logger.warning("PyPDF2 is not imported !!!")
        else:
            self.logger.info("PyPDF2 imported !!")

    def __read_pdf_file(self, file_name: str) -> list[str]:
        text = []
        try:
            with open(file_name, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                    except Exception as e:
                        self.logger.warning(
                            f"Error extracting page {page_num} from {file_name}: {str(e)}"
                        )
                        continue

            return "\n".join(text)

        except Exception as e:
            self.logger.error(f"Error reading PDF {file_name}: {str(e)}")
            raise

        return text

    def read_document(self, file_name: str) -> list[str]:
        """
        Returns contents of PDF file
        """
        try:
            content = self.__read_pdf_file(file_name)

            if content is None:
                self.logger.error(f"Unable to read file: {file_name}")
            
            return content
        except Exception as e:
            self.logger.error(f"Error reading PDF file: {file_name}: {e}")
        
        return None


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

for c in contents: print(c['metadata'])