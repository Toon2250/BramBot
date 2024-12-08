import streamlit as st
import os
import PyPDF2
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def initialize_chromadb():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = PersistentClient(
        path="./vector_db",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )

    collection = client.get_or_create_collection(name="pdf_documents")
    return collection, embedding_model

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def tokenize_and_store(collection, embedding_model, pdf_text, doc_id):
    embeddings = embedding_model.encode([pdf_text]).tolist()  # Generate embedding
    collection.add(
        documents=[pdf_text],
        metadatas=[{"source": "uploaded_pdf"}],
        ids=[doc_id],
        embeddings=embeddings
    )
    return "PDF content has been tokenized and stored successfully!"

# app
st.title("💬 BramBot - Upload zone")
st.write(
    "Here we upload the Pdf's so we can use them later for search-queries."
)

try:
    collection, embedding_model = initialize_chromadb()

    uploaded_file = st.file_uploader("Upload a PDF-file", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing and tokenizing PDF..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                doc_id = os.path.splitext(uploaded_file.name)[0]  # Use file name as ID, splitext removes the extension from te name
                result = tokenize_and_store(collection, embedding_model, pdf_text, doc_id)
                st.success(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
except Exception as exception:
    st.write(f"something went wrong: {exception}")