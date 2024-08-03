from fastapi import Request, HTTPException
from typing import Dict, Any
import re


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
