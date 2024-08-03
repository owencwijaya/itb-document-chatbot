from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from service.chain import create_chain
from service.utils import per_req_config_modifier


def create_app():
    app = FastAPI(title="Chatbot API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    chain = create_chain()

    add_routes(
        app, chain, path="/chatbot", per_req_config_modifier=per_req_config_modifier
    )

    return app
