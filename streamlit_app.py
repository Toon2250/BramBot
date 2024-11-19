import streamlit as st
from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = str(os.getenv("GROQ_API_KEY"))
# Initialize Groq client
client = Groq()
 
# Show title and description
st.title("💬 BramBot")
st.write(
    "This is a chatbot that uses Groq's powerful models to generate responses. "
    "To use this app, simply enter your message below and get an instant response from Groq's AI."
)
 
# Create a session state variable to store the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# Create an input field for user input
user_input = st.chat_input("What do you want to ask the bot?")
 
if user_input:
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    # Set up Groq Chat Completion
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant called BramBot."},
                {"role": "user", "content": user_input},
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            stream=False,  # Set to False to get the full response at once
        )
        # Extract and display assistant's response
        assistant_response = chat_completion.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
 
        # Display assistant's message
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
 
    except Exception as e:
        st.error(f"Error while getting response from Groq: {e}")