import streamlit as st
import fitz  # PyMuPDF for PDF processing
from transformers import pipeline

# Title of the App
st.title("PDF-Based Chatbot")
st.write("Upload a PDF, ask a question, and get an AI-powered answer!")

# File Upload Section
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

@st.cache_data
def extract_text_from_pdf(file):
    """
    Extracts text from an uploaded PDF file.
    """
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

@st.cache_resource
def load_qa_model():
    """
    Loads the question-answering model.
    """
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

if uploaded_file:
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("PDF uploaded and text extracted successfully!")

    # Load QA model
    qa_model = load_qa_model()

    # Input for user question
    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Get the answer
        with st.spinner("Finding the answer..."):
            result = qa_model(question=question, context=pdf_text)
            answer = result["answer"]
        st.write(f"**Answer:** {answer}")
