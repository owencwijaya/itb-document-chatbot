from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

from service.chain import create_chain

history = StreamlitChatMessageHistory(key="chatbot_demo")
chain = create_chain(get_session_history=lambda x: history)

st.title("Chatbot Demo")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("Masukkan pertanyaan:"):
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

        st.markdown(response["answer"])
