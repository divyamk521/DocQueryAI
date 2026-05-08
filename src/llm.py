from langchain_groq import ChatGroq
import config



def get_llm():
    """
    Load and return Groq model

    """

    llm=ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )
  
    return llm
    