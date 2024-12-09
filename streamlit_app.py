import streamlit as st
import importlib.util
import os

# Initialize session state variables
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""  # Initialize selected model in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""  # Initialize API key in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history

# Supported Model Providers (ensure this part exists in your code)
MODEL_PROVIDERS = {
    "Groq Llama 3 (70B)": {
        "model": "groq/llama3-70b-8192",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "Groq Gemma 2": {
        "model": "groq/gemma2-9b-it",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "Groq Mistral": {
        "model": "mixtral-8x7b-32768",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
}

# Title and description
st.title("💬 BramBot with Model Selector")
st.write(
    "This chatbot allows you to choose between powerful AI models from Groq, OpenAI, and Google. "
    "Enter your API key and select the model to begin chatting!"
)

# Model Selector
selected_model = st.selectbox(
    "Choose a Model Provider:", list(MODEL_PROVIDERS.keys()), index=0
)

# Reset the API key and chat history if a new model is selected
if st.session_state.selected_model != selected_model:
    st.session_state.api_key = ""  # Reset API key
    st.session_state.messages = []  # Reset chat history

st.session_state.selected_model = selected_model  # Update the selected model in session state

# Show API Key input only after a model is selected
if selected_model:
    st.session_state.api_key = st.text_input(
        "Enter your API Key:",  # Input prompt
        value=st.session_state.api_key or "",  # Pre-fill if previously entered
        type="password",  # Hide the API key for security
        placeholder="Your API Key here"  # Placeholder for guidance
    )

# Validate Input
if st.session_state.api_key:
    st.success(f"API Key provided! Selected model: {st.session_state.selected_model}")
    
    # Dynamically load the crew_ai_app module
    spec = importlib.util.spec_from_file_location("crew_ai_app", "crew_ai_app.py")
    crew_ai_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crew_ai_module)

    # Call the Crew AI function with selected model and API key
    if hasattr(crew_ai_module, "run_crew_ai_app"):
        selected_model_config = MODEL_PROVIDERS[st.session_state.selected_model]
        crew_ai_module.run_crew_ai_app(
            api_key=st.session_state.api_key,
            model_config=selected_model_config,
        )
else:
    st.warning("Please enter your API key to proceed.")