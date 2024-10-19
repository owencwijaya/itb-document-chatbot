from langchain_groq import ChatGroq
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_history_aware_retriever

from service.prompt import create_retrieval_prompt
from utils.settings import settings


model_name = {"embeddings": "BAAI/bge-base-en-v1.5", "llm": "llama3-8b-8192"}
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": settings.MILVUS_URI, "token": settings.MILVUS_TOKEN},
    collection_name="informasi_umum_json_refs",
)

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.25,
)

llama3_1_70b = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.25,
)

retriever = create_history_aware_retriever(
    llm, vectorstore.as_retriever(), create_retrieval_prompt()
)

db = SQLDatabase.from_uri(settings.DB_URI)
