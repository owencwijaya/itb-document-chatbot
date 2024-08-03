from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_qdrant import Qdrant
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from typing_extensions import TypedDict
from qdrant_client import QdrantClient

from utils.settings import settings
from service.prompt import create_rag_prompt, create_retrieval_prompt
from service.chat_history import create_chat_factory, history_factory_config


class InputChat(TypedDict):
    input: str


def init_models():
    model_name = {"embeddings": "BAAI/bge-base-en-v1.5", "llm": "llama-3.1-8b-instant"}

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        model_name=model_name["embeddings"], api_key=settings.HUGGINGFACE_API_KEY
    )

    vector_db_client = QdrantClient(settings.VECTOR_DB_ENDPOINT)

    llm = ChatGroq(
        model=model_name["llm"],
    )

    vectorstore = Qdrant(
        client=vector_db_client, collection_name="informasi_umum", embeddings=embeddings
    )

    retrieval_prompt = create_retrieval_prompt()

    retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), retrieval_prompt
    )

    return llm, retriever


def create_chain():
    llm, retriever = init_models()
    rag_prompt = create_rag_prompt()

    doc_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    print("before history chain")
    history_chain = RunnableWithMessageHistory(
        rag_chain,
        create_chat_factory(),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        history_factory_config=history_factory_config,
    ).with_types(input_type=InputChat)

    return history_chain
