import openai
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import config
import os
from parameters import chroma_persist_directory, dataset_dir

# connection strings
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_DEPLOYMENT_ENDPOINT = config.OPENAI_DEPLOYMENT_ENDPOINT
OPENAI_DEPLOYMENT_VERSION = config.OPENAI_DEPLOYMENT_VERSION
OPENAI_DEPLOYMENT_NAME = config.OPENAI_DEPLOYMENT_NAME
OPENAI_MODEL_NAME = config.OPENAI_MODEL_NAME
OPENAI_EMBEDDING_MODEL_NAME = config.OPENAI_EMBEDDING_MODEL_NAME
OPENAI_EMBEDDING_DEPLOYMENT_NAME = config.OPENAI_EMBEDDING_DEPLOYMENT_NAME
OPENAI_API_TYPE = config.OPENAI_API_TYPE

openai.api_key = OPENAI_API_KEY
# openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_version = OPENAI_DEPLOYMENT_VERSION

# Initialize llm
llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      openai_api_key=OPENAI_API_KEY,
                      azure_endpoint= OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      temperature=0.1)

# Initialize embeddings
embeddings_function = AzureOpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME,
                                   deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                                   openai_api_key=openai.api_key,
                                   azure_endpoint= OPENAI_DEPLOYMENT_ENDPOINT,
                                   openai_api_type=OPENAI_API_TYPE,
                                   openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                                   chunk_size=1
                                   )

