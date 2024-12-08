import streamlit as st
import os

UPLOAD_FOLDER = "./pdfs"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# app
st.title("💬 BramBot - Upload zone")
st.write(
    "Here we upload the Pdf's so we can use them later for search-queries."
)
pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]

if pdf_files:
    # Dropdown to select a PDF file
    selected_file = st.selectbox("Select a PDF file to view or process", pdf_files)
    if selected_file:
        st.write(f"You selected: {selected_file}")

try:
    uploaded_file = st.file_uploader("Upload a PDF-file", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Saving PDF..."):

            save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Saved PDF: {uploaded_file.name}")
except Exception as exception:
    st.error(f"Something went wrong: {exception}")