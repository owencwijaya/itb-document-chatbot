from service.chain import create_chain
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import uuid
import os

load_dotenv("../.env")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain = create_chain(get_session_history=get_session_history)


evaluators = [
    LangChainStringEvaluator("cot_qa"),
    LangChainStringEvaluator(
        "embedding_distance", config={"distance_metric": "cosine"}
    ),
    LangChainStringEvaluator(
        "labeled_score_string",
        config={
            "criteria": {
                "accuracy": "How accurate is this prediction compared to the reference on a scale of 1-10?"
            },
            "normalize_by": 10,
            "llm": ChatGroq(temperature=0.0, model="llama-3.1-8b-instant"),
        },
    ),
]


def custom_invoke(input_data):
    answer = chain.invoke(
        {"input": input_data["Pertanyaan"]},
        config={"configurable": {"user_id": "owen-" + str(uuid.uuid4())}},
    )

    return answer["answer"]


results = evaluate(
    custom_invoke,
    data="qna-informasi-umum",
    experiment_prefix="with-evaluators",
    max_concurrency=1,
    evaluators=evaluators,
)
