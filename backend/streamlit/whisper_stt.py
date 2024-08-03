from streamlit_mic_recorder import mic_recorder
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

import io
import os


def whisper_stt(
    groq_api_key=None,
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    language=None,
    callback=None,
    args=(),
    kwargs=None,
    key=None,
):
    if "client" not in st.session_state:
        st.session_state.client = Groq(api_key=groq_api_key)

    if not "_last_speech_to_text_transcript_id" in st.session_state:
        st.session_state._last_speech_to_text_transcript_id = 0
    if not "_last_speech_to_text_transcript" in st.session_state:
        st.session_state._last_speech_to_text_transcript = None
    if key and not key + "_output" in st.session_state:
        st.session_state[key + "_output"] = None

    audio = mic_recorder(
        start_prompt=start_prompt,
        stop_prompt=stop_prompt,
        just_once=just_once,
        use_container_width=use_container_width,
        format="webm",
        key=key,
    )

    new_output = False
    if audio is None:
        output = None
    else:
        id = audio["id"]
        new_output = id > st.session_state._last_speech_to_text_transcript_id
        if new_output:
            output = None
            st.session_state._last_speech_to_text_transcript_id = id
            audio_bio = io.BytesIO(audio["bytes"])
            audio_bio.name = "audio.webm"
            success = False
            err = 0

            while not success and err < 3:
                try:
                    transcript = st.session_state.client.audio.transcriptions.create(
                        model="whisper-large-v3", file=audio_bio, language=language
                    )
                except Exception as e:
                    print(str(e))
                    err += 1
                else:
                    success = True
                    output = transcript.text
                    st.session_state._last_speech_to_text_transcript = output
        elif not just_once:
            output = st.session_state._last_speech_to_text_transcript
        else:
            output = None

    if key:
        st.session_state[key + "_output"] = output
    if new_output and callback:
        callback(*args, **(kwargs or {}))
    return output
