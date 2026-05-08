from langchain_chroma import Chroma
import config

def create_vector_store(documents,embedding_model):

    """Create a ChromaDB store from a list of documents chunks.
    Use this during ingestion to save documents into the vector store.
    """
    store=Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )
    print("chromadb created successfully")
    return store

def get_vector_store(embedding_model):
    """
    Load exixting ChromaDB store from disk.
    """
    
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_DIR,
        embedding_function=embedding_model
        
    )
    

   

