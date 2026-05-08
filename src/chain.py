from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_classic.chains import create_retrieval_chain


def build_rag_chain(llm, prompt, retriever):
    """
    Build the complete RAG pipeline.
    """

    # Retrieved docs + question → prompt → LLM
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Retrieval step + generation step
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain