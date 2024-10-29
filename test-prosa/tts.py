import asyncio
import json
import os

import websockets

from dotenv import load_dotenv

load_dotenv()

PROSA_TTS_API_KEY = os.getenv("PROSA_TTS_API_KEY")


async def text_to_speech(
    texts: list[str],
    label: str = "test streaming",
    model_name: str = "tts-dimas-formal",
):
    url = "wss://tts-api.stg.prosa.ai/v2/speech/tts/streaming"
    fmt = "wav"
    sample_rate = 8000

    headers = {}

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"token": PROSA_TTS_API_KEY}))

        config = {
            "label": label,
            "model": model_name,
            "audio_format": fmt,
            "sample_rate": sample_rate,
        }

        await ws.send(json.dumps(config))

        data = json.loads(await ws.recv())

        job_id = data["id"]

        os.makedirs(f"results/{job_id}", exist_ok=True)

        try:
            for i, text in enumerate(texts):
                audio_name = f"audio{i}.{fmt}"
                print(audio_name)
                await ws.send(json.dumps({"text": text}))

                synthesized_audio = await ws.recv()

                assert isinstance(
                    synthesized_audio, bytes
                ), f"{audio_name}: No audio received"

                with open(f"results/{job_id}/{audio_name}", "wb") as file:
                    file.write(synthesized_audio)

        except websockets.exceptions.ConnectionClosedOK:
            print("WebSocket connection closed normally")
        except Exception as e:
            print(f"An error occurred: {e}")

    print(f"Text-to-speech operation completed. job_id: {job_id}")


# to test the function
# asyncio.run(text_to_speech(["Halo, perkenalkan nama saya Dimas"]))
