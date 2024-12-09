import streamlit as st
import pdfplumber
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize Qdrant and embedding model
qdrant = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('llama3-70b-8192')

COLLECTION_NAME = "pdf_chunks"

def embed_text(text):
    return model.encode(text).tolist()

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

st.title("Upload and Process PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.write("Processing the uploaded PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Split text into smaller chunks
    chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
    embeddings = [embed_text(chunk) for chunk in chunks]
    
    # Store embeddings in Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            {"id": f"chunk_{i}", "vector": embedding, "payload": {"text": chunk}}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
    )
    st.success("PDF processed and data stored in Qdrant!")
