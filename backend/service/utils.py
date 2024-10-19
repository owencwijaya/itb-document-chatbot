from xml.dom.minidom import Document
from fastapi import Request, HTTPException
from typing import Dict, Any
import re

from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnableSequence
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from service.prompt import SQL_ANSWER_PROMPT, create_sql_prompt
from service.models import llama3_1_70b, retriever, db, llm


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def is_valid_identifier(value: str) -> bool:
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def per_req_config_modifier(config: Dict[str, Any], request: Request) -> Dict[str, Any]:
    config = config.copy()
    configurable = config.get("configurable", {})
    user_id = request.cookies.get("user_id", None) or configurable.get("user_id", None)

    if user_id is None:
        raise HTTPException(status_code=400, detail="User ID not found!")

    configurable["user_id"] = user_id
    config["configurable"] = configurable

    return config


def document_retriever(query):
    if isinstance(query, str):
        query = {"input": query, "chat_history": []}

    results = retriever.invoke(query)
    documents = [
        Document(page_content=text, metadata={}) if isinstance(text, str) else text
        for text in results
    ]
    return documents


def sql_retriever(query):
    sql_query_chain = create_sql_query_chain(
        llama3_1_70b, db, k=10, prompt=create_sql_prompt()
    )

    context = db.get_context()
    print(list(context))
    print(context["table_info"])

    execute_query = QuerySQLDataBaseTool(db=db)
    # sql_chain = sql_query_chain | execute_query

    def print_query(inputs):
        print(f"DEBUG: Generated SQL Query: {inputs['query']}")
        print(f"DEBUG: SQL Query Result: {inputs['result']}")
        return inputs

    try:
        sql_runnable = RunnablePassthrough.assign(query=sql_query_chain).assign(
            result=itemgetter("query") | execute_query
        )

        sql_answer_chain = RunnableSequence(
            {
                "question": lambda x: x["question"],
                "query": lambda x: x["query"],
                "result": lambda x: x["result"],
            },
            print_query,
            SQL_ANSWER_PROMPT,
            llm,
            StrOutputParser(),
        )

        sql_chain = sql_runnable | sql_answer_chain

        answer = sql_chain.invoke({"question": query})
        print(f"DEBUG: sql answer: {answer}")
        return answer
    except Exception as e:
        print(f"DEBUG: Error in sql_retriever: {e}")
        raise e
