from document_loader import load_pdf, split_into_chunks
from embeddings import get_embedding_model
from vector_store import create_vector_store

FILE_PATH = r"D:\DocQueryAI\data\Health.pdf"

def run_ingestion():
        #step1:Loading my doc
        pages=load_pdf(FILE_PATH)

        print(f"Pages Loaded: {len(pages)}")

        #step2:Splitting into chunks
        chunks=split_into_chunks(pages)

        print(f"Created {len(chunks)} chunks")

        #step3:giving it to embedding model
        embedding_model=get_embedding_model()

        print("Embedding model loaded")

        #step4:creating vector store
        create_vector_store(chunks,embedding_model)

        print("Ingestion is complete now")
   

if __name__=="__main__":
        run_ingestion()
        

