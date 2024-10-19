from pydantic_settings import BaseSettings
from dotenv import load_dotenv

import os

load_dotenv("../.env")


class Settings(BaseSettings):
    VECTOR_DB_ENDPOINT: str = os.getenv("VECTOR_DB_ENDPOINT")
    CHAT_HISTORY_ENDPOINT: str = os.getenv("CHAT_HISTORY_ENDPOINT")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    MILVUS_URI: str = os.getenv("MILVUS_URI", "")
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    DB_URI: str = os.getenv("DB_URI", "")


settings = Settings()
