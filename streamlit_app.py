import streamlit as st
import importlib.util

# Initialize session state variables
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""  # Initialize selected model in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""  # Initialize API key in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history

COLLECTION_NAME = "pdf_chunks"
SESSION_HISTORY = "session_history"

# Supported Model Providers (ensure this part exists in your code)
MODEL_PROVIDERS = {
    "Llama 3 (70B)": {
        "model": "groq/llama3-70b-8192",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "Gemma 2": {
        "model": "groq/gemma2-9b-it",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "Mistral": {
        "model": "mixtral-8x7b-32768",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
}

# Title and description
if "qdrant_key" not in st.session_state:
    st.session_state.qdrant_key = ""  # Initialize API key in session state

if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""  # Initialize API key in session state


# Title and app description
st.title("💬 BramBot")
st.write(
    "This chatbot allows you to choose between powerful AI models from Groq."
    "Enter your API key and select the model to begin chatting!"
)

# Create two columns: one for the model selector and one for the API key input
col1, col2 = st.columns([4, 8])  # This gives the first column 8/12 and the second 4/12 width

# Model Selector in the second column (4/12)
with col1:
    selected_model = st.selectbox(
        "Choose a Model Provider:", list(MODEL_PROVIDERS.keys()), index=0
    )

# API Key input in the first column (8/12)
with col2:
    if selected_model:
        st.session_state.api_key = st.text_input(
            (f"Enter your {st.session_state.selected_model} API Key:"),  # Input prompt
            value=st.session_state.api_key or "",  # Pre-fill if previously entered
            type="password",  # Hide the API key for security
            placeholder="Your API Key here"  # Placeholder for guidance
        )

#Place Checkbox here
use_docs = st.checkbox("Use documents")

if use_docs:
    col3, col4 = st.columns([6, 6])  # This gives the first column 6/12 and the second 6/12 width
    with col3:
        st.session_state.qdrant_url = st.text_input(
            "Enter your Qdrant URL:",
            value=st.session_state.qdrant_url or "",
            placeholder="Your Qdrant URL here"
        )
    with col4:
            st.session_state.qdrant_key = st.text_input(
            "Enter your Qdrant API Key:",
            value=st.session_state.qdrant_key or "",
            type="password",
            placeholder="Your Qdrant API Key here"
        )

# Reset the API key and chat history if a new model is selected
if st.session_state.selected_model != selected_model:
    st.session_state.api_key = ""  # Reset API key
    st.session_state.messages = []  # Reset chat history

st.session_state.selected_model = selected_model  # Update the selected model in session state

# Validate Input
if use_docs:
    if st.session_state.api_key and st.session_state.qdrant_key and st.session_state.qdrant_url:
        st.success(f"API Key provided! Selected model: {st.session_state.selected_model}")
        spec = importlib.util.spec_from_file_location("crew_ai_app", "crew_ai_app.py")
        crew_ai_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crew_ai_module)

        # Call the Crew AI function with selected model and API key
        if hasattr(crew_ai_module, "run_crew_ai_app"):
            selected_model_config = MODEL_PROVIDERS[st.session_state.selected_model]
            crew_ai_module.run_crew_ai_app(
                api_key= st.session_state.api_key,
                qdrant_key= st.session_state.qdrant_key,
                qdrant_url= st.session_state.qdrant_url,
                model_config= selected_model_config,
                use_docs = use_docs,
            )
    else:
        st.warning("Please enter your API key to proceed.")
else:
    if st.session_state.api_key:
        st.success(f"API Key provided! Selected model: {st.session_state.selected_model}")
        spec = importlib.util.spec_from_file_location("crew_ai_app", "crew_ai_app.py")
        crew_ai_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crew_ai_module)

        # Call the Crew AI function with selected model and API key
        if hasattr(crew_ai_module, "run_crew_ai_app"):
            selected_model_config = MODEL_PROVIDERS[st.session_state.selected_model]
            crew_ai_module.run_crew_ai_app(
                api_key= st.session_state.api_key,
                qdrant_key= st.session_state.qdrant_key,
                qdrant_url= st.session_state.qdrant_url,
                model_config= selected_model_config,
                use_docs = use_docs,
            )
    else:
        st.warning("Please enter your API key to proceed.")