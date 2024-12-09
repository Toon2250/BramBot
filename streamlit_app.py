import streamlit as st
import importlib.util
import openai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "pdf_chunks"
SESSION_HISTORY = "session_history"

if "api_key" not in st.session_state:
    st.session_state.api_key = ""  # Initialize API key in session state

if "qdrant_key" not in st.session_state:
    st.session_state.qdrant_key = ""  # Initialize API key in session state

if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""  # Initialize API key in session state

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""  # Initialize API key in session state

if SESSION_HISTORY not in st.session_state:
    st.session_state[SESSION_HISTORY] = {"Default": []}


# Title and app description
st.title("💬 BramBot")
st.write(
    "This is a chatbot that uses Groq's powerful models to generate responses. "
    "To use this app, simply enter your message below and get an instant response from Groq's AI."
)
history_sessions = list(st.session_state[SESSION_HISTORY].keys())
selected_session = st.selectbox("Select a session:", history_sessions)

# Create a new session
if st.button("Start New Session"):
    new_session_name = f"Session {len(history_sessions) + 1}"
    st.session_state[SESSION_HISTORY][new_session_name] = []
    st.experimental_rerun()

st.session_state.api_key = st.text_input(
    "Enter your Groq API Key:",  # Input prompt
    value=st.session_state.api_key or "",  # Pre-fill if previously entered
    type="password",  # Hide input for security
    placeholder="Your API Key here"  # Placeholder for guidance
)

st.session_state.qdrant_key = st.text_input(
    "Enter your Qdrant API Key:",  # Input prompt
    value=st.session_state.qdrant_key or "",  # Pre-fill if previously entered
    type="password",  # Hide input for security
    placeholder="Your API Key here"  # Placeholder for guidance
)

st.session_state.qdrant_url = st.text_input(
    "Enter your Qdrant API Key:",  # Input prompt
    value=st.session_state.qdrant_url or "",  # Pre-fill if previously entered
    type="default",  # Hide input for security
    placeholder="Your API Key here"  # Placeholder for guidance
)

st.session_state.openai_key = st.text_input(
    "Enter your OpenAI API Key:",  # Input prompt
    value=st.session_state.openai_key or "",  # Pre-fill if previously entered
    type="password",  # Hide input for security
    placeholder="Your API Key here"  # Placeholder for guidance
)

if st.session_state.api_key:
    st.success("API Key provided! Loading the chatbot application...")


    # Dynamically load the second file
    spec = importlib.util.spec_from_file_location("crew_ai_app", "crew_ai_app.py")
    crew_ai_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crew_ai_module)

    # Call a function from the loaded module (if defined in `crew_ai_app.py`)
    if hasattr(crew_ai_module, "run_crew_ai_app"):
        crew_ai_module.run_crew_ai_app(st.session_state.api_key, st.session_state.qdrant_key, st.session_state.qdrant_url, st.session_state.openai_key)
else:
    st.warning("Please enter your API key to proceed.")

