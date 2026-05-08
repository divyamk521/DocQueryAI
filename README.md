# DocQueryAI

An end-to-end Retrieval-Augmented Generation (RAG) application built using FastAPI-ready modular architecture, LangChain, ChromaDB, HuggingFace Embeddings, and Groq LLMs.

DocQueryAI allows users to ingest PDF documents, convert them into vector embeddings, store them in ChromaDB, and perform intelligent semantic question answering over the uploaded documents.

---

# Features

* Modular RAG architecture
* PDF ingestion pipeline
* Semantic chunking using LangChain text splitters
* Vector embeddings with HuggingFace models
* ChromaDB vector storage
* Semantic retrieval using retrievers
* Groq-powered LLM responses
* Retrieval debugging support
* Production-oriented project structure
* Easy to extend with FastAPI or Streamlit

---

# Tech Stack

| Technology             | Purpose                         |
| ---------------------- | ------------------------------- |
| Python                 | Core programming language       |
| LangChain              | RAG orchestration               |
| ChromaDB               | Vector database                 |
| HuggingFace Embeddings | Text embeddings                 |
| Groq API               | LLM inference                   |
| dotenv                 | Environment variable management |

---

# Project Architecture

```text
PDF Documents
      ↓
Document Loader
      ↓
Chunking Pipeline
      ↓
Embedding Model
      ↓
ChromaDB Vector Store
      ↓
Retriever
      ↓
Prompt Template
      ↓
Groq LLM
      ↓
Final AI Answer
```

---


# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd DocQueryAI
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the root directory.

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Configuration

The project uses a centralized `config.py` file.

```python
# LLM settings
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ChromaDB
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "documents"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Retrieval
RETRIEVER_K = 4
```

---

# How the System Works

## Step 1 — Document Loading

PDF documents are loaded using LangChain document loaders.

```python
pages = load_pdf(FILE_PATH)
```

---

## Step 2 — Chunking

Documents are split into smaller semantic chunks.

```python
chunks = split_into_chunks(pages)
```

---

## Step 3 — Embedding Generation

Chunks are converted into dense vector embeddings.

```python
embedding_model = get_embedding_model()
```

---

## Step 4 — Vector Storage

Embeddings are stored in ChromaDB.

```python
create_vector_store(chunks, embedding_model)
```

---

## Step 5 — Retrieval

Retriever performs semantic similarity search.

```python
retriever.invoke(question)
```

---

## Step 6 — RAG Generation

Relevant chunks are injected into the prompt and sent to the LLM.

```python
response = rag_chain.invoke({
    "input": question
})
```

---

# Running the Project

## Step 1 — Ingest Documents

Run ingestion to create embeddings and store vectors.

```bash
python ingest.py
```

---

## Step 2 — Start Retrieval System

```bash
python retrieval.py
```

---

# Example

```text
You: What are the symptoms of diabetes?

AI Answer:
Common symptoms of diabetes include increased thirst,
frequent urination, fatigue, blurred vision, and slow healing wounds.
```

---

# Retrieval Debugging

The project includes retrieval inspection debugging.

Example:

```python
print(docs[0].page_content)
```

This helps analyze:

* Retrieval quality
* Chunk relevance
* Embedding effectiveness
* Hallucination sources

---

# Production Improvements

Potential future improvements:

* FastAPI backend integration
* Streamlit frontend
* Multi-PDF ingestion
* Metadata filtering
* Hybrid search
* Re-ranking pipelines
* Streaming responses
* Chat history memory
* Source citations
* Docker deployment

---

# Why This Project Matters

This project demonstrates:

* Real-world RAG architecture
* Semantic search systems
* Vector databases
* LLM orchestration
* AI backend engineering fundamentals
* Modular software engineering practices

---

# Learning Outcomes

By building this project, you understand:

* How embeddings work
* How vector similarity search works
* How retrieval impacts LLM quality
* How LangChain chains operate internally
* How RAG systems are engineered in production

---

# Author

Divya M Kumar

---

# License

This project is open-source and available under the MIT License.
