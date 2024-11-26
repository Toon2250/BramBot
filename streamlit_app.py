import streamlit as st
from groq import Groq
import os
import traceback

# Title and app description
st.title("💬 BramBot")
st.write(
    "This is a chatbot that uses Groq's powerful models to generate responses. "
    "To use this app, simply enter your message below and get an instant response from Groq's AI."
)

# Step 1: Input field for the Groq API key
if "api_key" not in st.session_state:
    st.session_state.api_key = None  # Initialize API key in session state

st.session_state.api_key = st.text_input(
    "Enter your Groq API Key:",  # Input prompt
    value=st.session_state.api_key or "",  # Pre-fill if previously entered
    type="password",  # Hide input for security
    placeholder="Your API Key here"  # Placeholder for guidance
)

# Validate the API key input
if st.session_state.api_key:
    # Step 2: Clean the API key (remove leading/trailing spaces)
    api_key = st.session_state.api_key.strip()

    # Store the cleaned API key in environment variables (required by Groq)
    os.environ["GROQ_API_KEY"] = api_key

    # Step 3: Attempt to initialize the Groq client
    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])  # Initialize Groq client
        st.success("Groq client initialized successfully!")  # Success message
    except Exception as e:
        # Display error and debugging info if client initialization fails
        st.error(f"Error initializing Groq client: {e}")
        st.write("Debug Info:", {
            "API Key Present": bool(os.environ.get("GROQ_API_KEY")),  # Verify API key exists
            "API Key Value": os.environ.get("GROQ_API_KEY")[:4] + "..." if os.environ.get("GROQ_API_KEY") else None,  # Show partial API key for verification
            "Traceback": traceback.format_exc()  # Detailed error traceback
        })
        st.stop()  # Stop further execution if initialization fails
else:
    # Show a warning and halt the app if no API key is provided
    st.warning("Please enter your Groq API key to use the chatbot.")
    st.stop()

# Step 4: Test connectivity to Groq API
try:
    # Send a simple request to verify connectivity
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": "Test connection"}],  # Test message
        model="llama3-70b-8192",  # Specify the model
        temperature=0.1,  # Low randomness for testing
        max_tokens=10,  # Minimal response for a quick check
    )
    st.success("Connected to Groq API successfully!")  # Success message
except Exception as e:
    # Handle connectivity errors and display debug info
    st.error(f"Connection failed: {e}")
    st.write("Debug Info:", {
        "API Key Present": bool(os.environ.get("GROQ_API_KEY")),  # Check API key presence
        "API Key Value": os.environ.get("GROQ_API_KEY")[:4] + "..." if os.environ.get("GROQ_API_KEY") else None,  # Partial API key
        "Error": str(e),  # Error message
        "Traceback": traceback.format_exc()  # Full error traceback
    })
    st.stop()  # Stop further execution if connection fails

# Step 5: Chatbot functionality
# Initialize the chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores user and bot messages

# Display previous chat messages in the app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Display messages as user or assistant
        st.markdown(message["content"])

# Input field for user messages
user_input = st.chat_input("What do you want to ask the bot?")  # Input box for user queries

if user_input:
    # Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):  # Display user's message
        st.markdown(user_input)

    # Step 6: Generate a response from Groq's API
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant called BramBot, and you are very passionate about Artificial intelligence, Deep Learning, Natural language processing."},  # System instructions
                {"role": "user", "content": user_input},  # User's message
            ],
            model="llama3-70b-8192",  # Specify the model to use
            temperature=0.5,  # Adjust randomness for creativity
            max_tokens=1024,  # Limit response length
            stream=False,  # Get the full response at once
        )

        # Extract the assistant's response from the API
        assistant_response = chat_completion.choices[0].message.content

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Display the assistant's message
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    except Exception as e:
        # Handle errors during response generation
        st.error(f"Error while getting response from Groq: {e}")
        st.write("Debug Info:", {
            "User Input": user_input,  # User's query for context
            "API Key": os.environ.get("GROQ_API_KEY")[:4] + "..." if os.environ.get("GROQ_API_KEY") else None,  # Partial API key
            "Error": str(e),  # Error message
            "Traceback": traceback.format_exc()  # Detailed traceback
        })
