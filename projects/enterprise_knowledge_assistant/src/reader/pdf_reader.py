import logging
from typing import Optional

try:
    import PyPDF2
except Exception as e:
    print("Cannot import PyPDF2: ", e)


class PDFReader:

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
                metadata = pdf_reader.metadata
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

            # return "\n".join(text)
            return {
                "content": "\n".join(text),
                "metadata": metadata
            }

        except Exception as e:
            self.logger.error(f"Error reading PDF {file_name}: {str(e)}")
            raise

        return text

    # def read_document(self, file_name: str) -> list[str]:
    def read_document(self, file_name: str) -> dict:
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
