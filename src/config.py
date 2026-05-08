from dotenv import load_dotenv
import os

load_dotenv()

# LLM settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "documents"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Retrieval
RETRIEVER_K = 4