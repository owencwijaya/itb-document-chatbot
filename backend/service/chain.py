from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict

from service.utils import document_retriever, format_docs, sql_retriever
from service.prompt import create_rag_prompt
from service.chat_history import create_chat_factory, history_factory_config
from service.models import llm


class InputChat(TypedDict):
    input: str


def create_chain(get_session_history=create_chat_factory()):

    rag_prompt = create_rag_prompt()

    combined_retriever = RunnablePassthrough.assign(
        context=lambda x: document_retriever(x),
        sql_result=lambda x: sql_retriever(x["input"]),
    )

    rag_chain = RunnableSequence(
        {
            "context": lambda x: format_docs(x["context"]),
            "sql_result": lambda x: x["sql_result"],
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | rag_prompt
        | llm
        | StrOutputParser(),
    )

    final_chain = combined_retriever | rag_chain

    history_chain = RunnableWithMessageHistory(
        final_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        history_factory_config=history_factory_config,
    ).with_types(input_type=InputChat)

    return history_chain
