import asyncio
import json
import os

import websockets

from dotenv import load_dotenv

load_dotenv()

PROSA_STT_API_KEY = os.getenv("PROSA_STT_API_KEY")


async def send_audio(
    filename: str, ws: websockets.WebSocketClientProtocol, chunk_size: int = 16000
):
    with open(filename, "rb") as file:
        while data := file.read(chunk_size):
            await ws.send(data)

        await ws.send(b"")


async def receive_message(ws: websockets.WebSocketClientProtocol):
    while True:
        data = json.loads(await ws.recv())

        message_type = data["type"]
        if message_type == "result":
            transcript = data["transcript"]
            return transcript


async def speech_to_text(filename: str) -> str:
    url = "wss://asr-api.stg.prosa.ai/v2/speech/stt/streaming"

    headers = {
        "x-api-key": PROSA_STT_API_KEY,
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        config = {
            "label": "Streaming STT Test",
            "model": "asr-general-online",
            "include_partial": False,
        }

        await ws.send(json.dumps(config))

        _, transcript = await asyncio.gather(
            send_audio(filename, ws), receive_message(ws)
        )

        return transcript


# asyncio.run(speech_to_text("audio/Recording 3.flac"))
