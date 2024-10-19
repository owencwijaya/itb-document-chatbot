from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.chat_history import BaseChatMessageHistory
from service.prompt import SQL_PROMPT, create_retrieval_prompt
from langchain.chains.sql_database.query import (
    create_sql_query_chain,
)
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


def retrieve_sql(
    query: str,
    chat_history: BaseChatMessageHistory,
    db: SQLDatabase,
    llm: BaseChatModel,
):
    # reformulate the query first
    prompt = create_retrieval_prompt(query, chat_history)

    reformulated_query = llm.invoke(prompt)

    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)

    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | SQL_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question": query, "query": reformulated_query})
