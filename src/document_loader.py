from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

FILE_PATH=r"D:\DocQueryAI\data\Health.pdf"


def load_pdf(file_path):
    """
    Load a pdf file and return a list of pages as LangChain documents.
    """
    loader=PyPDFLoader(file_path)
    pages=loader.load()
    return pages



def split_into_chunks(documents):
    """
    Split documents into smaller overlapping chunks.
    smaller chunks = better retrieval accuracy.
    """
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
          chunk_overlap=config.CHUNK_OVERLAP
          )
    chunks=splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    pdf_pages = load_pdf(FILE_PATH)

    chunks = split_into_chunks(pdf_pages)
    print(f"Pages Loaded: {len(pdf_pages)}")
    print(f"Chunks Created: {len(chunks)}")
 