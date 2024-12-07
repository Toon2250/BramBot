import streamlit as st
import os
import PyPDF2
from chromadb import Client
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def initialize_chromadb(openai_api_key):
    embedding_fn = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

    client = PersistentClient(
        path="./vector_db",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )

    collection = client.get_or_create_collection(name="pdf_documents", embedding_function=embedding_fn)
    return collection, embedding_fn

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def tokenize_and_store(collection, embedding_fn, pdf_text, doc_id):
    embeddings = embedding_fn([pdf_text])  # Generate embedding
    collection.add(
        documents=[pdf_text],
        metadatas=[{"source": "uploaded_pdf"}],
        ids=[doc_id]
    )
    return "PDF content has been tokenized and stored successfully!"

# app
st.title("💬 BramBot - Upload zone")
st.write(
    "Here we upload the Pdf's so we can use them later for search-queries."
)

try:
    openai_api_key = st.session_state.openai_api_key

    collection, embedding_fn = initialize_chromadb(openai_api_key)

    uploaded_file = st.file_uploader("Upload a PDF-file", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing and tokenizing PDF..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                doc_id = os.path.splitext(uploaded_file.name)[0]  # Use file name as ID, splitext removes the extension from te name
                result = tokenize_and_store(collection, embedding_fn, pdf_text, doc_id)
                st.success(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
except Exception as exception:
    st.write(f"something went wrong: {exception}")