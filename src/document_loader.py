import pymupdf4llm

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

import config


FILE_PATH = r"D:\DocQueryAI\data\Health.pdf"


def load_pdf(file_path):
    """
    Convert PDF into markdown text.
    """

    markdown_text = pymupdf4llm.to_markdown(file_path)

    return markdown_text


def split_into_chunks(markdown_text):
    """
    Split markdown into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.create_documents([markdown_text])

    return chunks


if __name__ == "__main__":

    markdown_text = load_pdf(FILE_PATH)

    chunks = split_into_chunks(markdown_text)

    print(f"Chunks Created: {len(chunks)}")

    print("\nFIRST CHUNK:\n")
    print(chunks[168].page_content)