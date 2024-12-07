import streamlit as st
import importlib.util

if "api_key" not in st.session_state:
    st.session_state.api_key = ""  # Initialize API key in session state

# Title and app description
st.title("💬 BramBot")
st.write(
    "This is a chatbot that uses Groq's powerful models to generate responses. "
    "To use this app, simply enter your message below and get an instant response from Groq's AI."
)

st.session_state.api_key = st.text_input(
    "Enter your Groq API Key:",  # Input prompt
    value=st.session_state.api_key or "",  # Pre-fill if previously entered
    type="password",  # Hide input for security
    placeholder="Your API Key here"  # Placeholder for guidance
)

# Step 5: Chatbot functionality
# Initialize the chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores user and bot messages

if st.session_state.api_key:
    st.success("API Key provided! Loading the chatbot application...")

    # Dynamically load the second file
    spec = importlib.util.spec_from_file_location("crew_ai_app", "crew_ai_app.py")
    crew_ai_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crew_ai_module)

    # Call a function from the loaded module (if defined in `crew_ai_app.py`)
    if hasattr(crew_ai_module, "run_crew_ai_app"):
        crew_ai_module.run_crew_ai_app(st.session_state.api_key)
else:
    st.warning("Please enter your API key to proceed.")