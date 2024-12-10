import streamlit as st
import pdfplumber
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize Qdrant API key and URL
if "qdrant_key" not in st.session_state:
    st.session_state.qdrant_key = ""  # Initialize API key in session state

if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""  # Initialize API key in session state

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

# Initialize SentenceTransformer model
ST_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Ensure Qdrant API key and URL are provided
if st.session_state.qdrant_key and st.session_state.qdrant_url:

    # Initialize Qdrant Client
    qdrant = QdrantClient(url=st.session_state.qdrant_url, api_key=st.session_state.qdrant_key, timeout=60)

    COLLECTION_NAME = "pdf_chunks"

    # Function to check if a collection exists and create it if not
    def create_collection_if_not_exists():
        try:
            # Try to retrieve collection info
            qdrant.get_collection(COLLECTION_NAME)
            st.write(f"Collection '{COLLECTION_NAME}' already exists.")
        except Exception:
            # If collection doesn't exist, create it
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 384, "distance": "Cosine"}  # Correct size based on model output
            )
            st.write(f"Collection '{COLLECTION_NAME}' created.")

    # Call function to ensure collection exists
    create_collection_if_not_exists()

    # Function to extract text from the uploaded PDF
    def extract_text_from_pdf(file):
        with pdfplumber.open(file) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

    # Function to get embeddings for the text chunks using Hugging Face model
    def get_embeddings(texts):
        # The ST_model.encode() will generate embeddings (vectors) for the input text
        return ST_model.encode(texts)

    # Streamlit UI to upload PDF and process it
    st.title("Upload and Process PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        st.write("Processing the uploaded PDF...")
        
        pdf_name = uploaded_file.name
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Split the PDF text into smaller chunks (adjust as needed)
        chunk_size = 500  # Adjust this based on your needs
        chunks = [pdf_text[i:i + chunk_size] for i in range(0, len(pdf_text), chunk_size)]
        
        # Get embeddings for each chunk using Hugging Face model
        embeddings = get_embeddings(chunks)
        
        # Store chunks with their corresponding vectors in Qdrant
        points = [
            {
                "id": i,
                "vector": embeddings[i].tolist(),  # Convert numpy array to list for Qdrant
                "payload": {"Source": pdf_name, "text": chunk}
            }
            for i, chunk in enumerate(chunks)
        ]

        batch_size = 100
        # Insert chunks with vectors into Qdrant
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

        st.session_state.file_uploader = None

        st.success("PDF "+ pdf_name +" processed and text with vectors stored in Qdrant!")

else:
    st.warning("Please enter your Qdrant API key and URL to proceed.")