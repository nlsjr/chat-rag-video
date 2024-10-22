import os

from langchain_chroma import Chroma

from utils_model import get_embedding_model


def get_filtered_vector_store(metadata_filter):
    vector_store = Chroma(
        collection_name="mm_rag_clip_photos",
        persist_directory=os.getcwd() + "/index",
        embedding_function=get_embedding_model()
    )

    vector_store.get(where=metadata_filter)

    return vector_store
