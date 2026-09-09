
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI


# -----------------------------
# Model
# -----------------------------

model = ChatMistralAI(
    model="ministral-3b-2512",
    temperature=0.7
)


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Mode Based Chatbot",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🤖 Mode Based Chatbot")
st.write("Choose the personality of your AI agent")


# -----------------------------
# Mode Selection
# -----------------------------

mode_options = {
    "😡 Angry Mode": "you are an angry AI agent. you are very rude and sarcastic. you will answer in a very angry way in short words",
    "😂 Funny Mode": "you are a funny AI agent. you will answer in a very funny way in short words",
    "😢 Sad Mode": "you are a sad AI agent. you will answer in a very sad way in short words"
}


selected_mode = st.radio(
    "Choose chatbot mode:",
    list(mode_options.keys())
)


# -----------------------------
# Initialize Chat History
# -----------------------------

if "messages" not in st.session_state:

    mode = mode_options[selected_mode]

    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

    st.session_state.current_mode = selected_mode


# -----------------------------
# Reset chat when mode changes
# -----------------------------

if st.session_state.current_mode != selected_mode:

    mode = mode_options[selected_mode]

    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

    st.session_state.current_mode = selected_mode


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.write(message.content)


# -----------------------------
# Chat Input
# -----------------------------

prompt = st.chat_input("Type your message...")


if prompt:

    # Keep original exit functionality
    if prompt == "0":
        st.stop()

    # Add user message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.write(prompt)


    # Invoke model using the same message history
    response = model.invoke(
        st.session_state.messages
    )


    # Add AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )


    # Display response
    with st.chat_message("assistant"):
        st.write(response.content)
