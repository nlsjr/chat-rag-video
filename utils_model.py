import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

_ = load_dotenv(find_dotenv())

openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION')

def get_llm():
    llm = AzureChatOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=api_key,
        azure_deployment="gpt-4o",
        api_version=api_version,
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    return llm


def get_embedding_model():
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=openai_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    return embedding_model
