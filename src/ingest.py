from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma




#Loading
file_path = "D:\DocQueryAI\data\Health.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
# print(docs[0].page_content)

#Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print("TOTAL CHUNKS:", len(chunks))

#Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully")

#Extracting text from chunks
chunk_texts = [chunk.page_content for chunk in chunks]
#converting text chunks into embeddings/vector
embeddings = embedding_model.embed_documents(chunk_texts)

print("\nFIRST CHUNK:\n")
print(chunk_texts[0])

print("\nFIRST 10 VALUES OF FIRST EMBEDDING:\n")
print(embeddings[0][:10])

#initializing Chroma vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("\nDocuments stored successfully in ChromaDB") 