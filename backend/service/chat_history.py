from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema.runnable.utils import ConfigurableFieldSpec

from typing import Callable

from service.utils import is_valid_identifier
from utils.settings import settings


def get_chat_history(user_id: str) -> BaseChatMessageHistory:
    if not is_valid_identifier(user_id):
        raise ValueError(f"Invalid user ID: {user_id}")

    chat_history = RedisChatMessageHistory(
        session_id=user_id, key_prefix="", url=settings.CHAT_HISTORY_ENDPOINT
    )

    return chat_history


def create_chat_factory() -> Callable[[str, str], BaseChatMessageHistory]:
    return get_chat_history


history_factory_config = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        default="",
        is_shared=True,
    ),
]
