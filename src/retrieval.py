from config import settings

from langchain_chroma import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_classic.chains import create_retrieval_chain

#loading embedding model for query
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully")

#connecting embedding to chroma
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

print("Connected to ChromaDB successfully")

#create retriver
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)

print("Retriever created successfully")

#initializing llm
llm = ChatGroq(
    groq_api_key=settings.GROQ_API_KEY,
    model_name="openai/gpt-oss-120b"
)

print("LLM initialized successfully")

#creating prompt template
prompt = ChatPromptTemplate.from_template(
    """
Answer the question only using the provided context.

<context>
{context}
</context>

Question:
{input}
"""
)

print("Prompt template created successfully")

#doc chain
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

print("Document chain created successfully")

#retrival chain
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

print("Retrieval chain created successfully")

#user query
query = input("whats the methodology ")


#running retrival chain
response = retrieval_chain.invoke({
    "input": query
})


#printing ans
print("\nANSWER:\n")

print(response["answer"])