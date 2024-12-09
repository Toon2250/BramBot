import streamlit as st
import importlib.util

COLLECTION_NAME = "pdf_chunks"
SESSION_HISTORY = "session_history"

# Initialize session state variables
if "api_key" not in st.session_state:
    st.session_state.api_key = ""  # Initialize API key in session state

if "qdrant_key" not in st.session_state:
    st.session_state.qdrant_key = ""  # Initialize API key in session state

if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""  # Initialize API key in session state

if SESSION_HISTORY not in st.session_state:
    st.session_state[SESSION_HISTORY] = {}

# Title and app description
st.title("💬 BramBot")
st.write(
    "This is a chatbot that uses Groq's powerful models to generate responses. "
    "To use this app, simply enter your message below and get an instant response from Groq's AI."
)

# API key inputs
st.session_state.api_key = st.text_input(
    "Enter your Groq API Key:",
    value=st.session_state.api_key or "",
    type="password",
    placeholder="Your API Key here"
)

st.session_state.qdrant_key = st.text_input(
    "Enter your Qdrant API Key:",
    value=st.session_state.qdrant_key or "",
    type="password",
    placeholder="Your Qdrant API Key here"
)

st.session_state.qdrant_url = st.text_input(
    "Enter your Qdrant URL:",
    value=st.session_state.qdrant_url or "",
    placeholder="Your Qdrant URL here"
)

# Ensure API key is provided
if st.session_state.api_key and st.session_state.qdrant_key and st.session_state.qdrant_url:
    st.success("API Key's provided! Loading the chatbot application...")
    
    # Dynamically load the second file
    spec = importlib.util.spec_from_file_location("crew_ai_app", "crew_ai_app.py")
    crew_ai_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crew_ai_module)

    # Call a function from the loaded module (if defined in `crew_ai_app.py`)
    if hasattr(crew_ai_module, "run_crew_ai_app"):
        crew_ai_module.run_crew_ai_app(
            st.session_state.api_key,
            st.session_state.qdrant_key,
            st.session_state.qdrant_url,
        )
        
else:
    st.warning("Please enter your API key to proceed.")