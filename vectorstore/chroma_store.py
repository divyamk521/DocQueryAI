from langchain_chroma import Chroma

from src.ingest import process_document


# PROCESS DOCUMENT
file_path = "data/Health.pdf"

chunks, embedding_model = process_document(file_path)


# STORE IN CHROMADB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Documents stored successfully in ChromaDB")