import asyncio
from langserve import RemoteRunnable
from stt import speech_to_text
from tts import text_to_speech

pipeline_url = "http://localhost:6900/chatbot"
chain = RemoteRunnable(pipeline_url)

filepath = "audio/Recording 3.flac"
transcript = asyncio.run(speech_to_text(filepath))

print(f"transcript: {transcript}")
response = chain.invoke(
    {"input": transcript},
    {
        "configurable": {
            "user_id": "test_user",
        }
    },
)

print(f"response: {response}")
asyncio.run(text_to_speech([response]))
