from langchain_huggingface import HuggingFaceEmbeddings
import config


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={'normalize_embeddings': True}
    )

    return embedding_model
