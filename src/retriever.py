from vector_store import get_vector_store
from embeddings import get_embedding_model
import config



def get_retriever():

    """
    Build and return a retriever that searches ChromaDB.
 
    The retriever takes a question, embeds it, and finds the
    top K most similar document chunks from the vector store.
    """

    embedding_model=get_embedding_model()
    store=get_vector_store(embedding_model)
    retriever=store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K}
    )
    
    return retriever
        
    