import asyncio
import os
from langserve import RemoteRunnable
from stt import speech_to_text
from tts import text_to_speech
import sounddevice as sd
import wave

pipeline_url = "http://localhost:6900/chatbot"
chain = RemoteRunnable(pipeline_url)

# Ensure the recordings directory exists
os.makedirs("recordings", exist_ok=True)


def record_audio(duration: int, filename: str) -> None:
    print("Recording...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished
    # Save the recorded audio to a file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())


def play_audio(file_path: str):
    with wave.open(file_path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        sd.play(data, samplerate=wf.getframerate())
        sd.wait()


async def main():
    while True:
        print("Press Enter to start recording...")
        input()
        filename = "recordings/recording.wav"  # Specify the file path
        record_audio(5, filename)  # Record for 5 seconds and save to file

        with open(filename, "rb") as audio_file:
            audio_bytes = audio_file.read()
            transcript = await speech_to_text(audio_bytes)  # Load from file
            print(f"Transcript: {transcript}")

        if transcript is None:
            print("Error: transcript is None")
            break

        response = chain.invoke(
            {"input": transcript},
            {
                "configurable": {
                    "user_id": "test_user",
                }
            },
        )

        print(f"Response: {response}")
        audio_file_path = await text_to_speech([response])
        play_audio(audio_file_path)

        user_input = input("Press Enter to continue or type 'exit' to stop: ")
        if user_input.lower() == "exit":
            break


# Run the main function
asyncio.run(main())
