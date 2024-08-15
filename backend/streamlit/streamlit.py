from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from whisper_stt import whisper_stt
from dotenv import load_dotenv

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.chain import create_chain

load_dotenv("../../.env")

history = StreamlitChatMessageHistory(key="chatbot_demo")
chain = create_chain(get_session_history=lambda x: history)

st.title("Chatbot Demo")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

text = whisper_stt(groq_api_key=os.getenv("GROQ_API_KEY"), language="id")

if prompt := st.chat_input("Masukkan pertanyaan:") or text:
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamlitCallbackHandler(st.container())
        with st.spinner("Sedang berpikir..."):
            response = chain.invoke(
                {
                    "input": prompt,
                },
                config={
                    "callbacks": [stream_handler],
                    "configurable": {"user_id": "owen"},
                },
            )
            # response = st.write_stream(
            #     chain.stream(
            #         {
            #             "input": prompt,
            #         },
            #         config={
            #             "callbacks": [stream_handler],
            #             "configurable": {"user_id": "owen"},
            #         },
            #     )
            # )

        st.markdown(response["answer"])
