from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    """
    Return the RAG prompt template.
    """

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question as best as you can using the provided context.

If the answer is not explicitly stated but can be inferred, do so.

Only if the context is completely unrelated should you say you don't have enough information.

Context:
{context}

Question:
{input}

Answer:
"""
    )

    return prompt