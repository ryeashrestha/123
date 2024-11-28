import streamlit as st
import fitz  # PyMuPDF for PDF processing
import google.generativeai as genai  # Google's Generative AI

GOOGLE_API_KEY = "AIzaSyDaDtYl04XGFzb5xmLgkuVBCSqn6JwkJlU"  # Removed actual API key for security
genai.configure(api_key=GOOGLE_API_KEY)

# Load the fine-tuned model (replace with your actual model ID)
model = genai.GenerativeModel('<MODEL_ID>')  # Removed actual model ID for security

# Title and description of the app
st.title("PDF-Based Chatbot with Google Gemini API")
st.write("Upload a PDF, ask a question, and get AI-powered answers based on the PDF content!")

# PDF file upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

@st.cache_data
def extract_text_from_pdf(file):
    """Extracts text from the uploaded PDF using PyMuPDF."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def chat_with_bot(pdf_text, question):
    """Send the PDF content and question to the generative AI model and get the answer."""
    full_prompt = f"Context from the PDF:\n{pdf_text}\n\nQuestion: {question}"
    
    # Create a chat session and get the response
    chat = model.start_chat(history=[])
    response = chat.send_message(full_prompt, stream=True)
    
    # Collect the response text
    response_text = ""
    for chunk in response:
        if chunk.text:
            response_text += chunk.text + ' '
    
    return response_text

if uploaded_file:
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("PDF uploaded and text extracted successfully!")

    # Show a preview of the extracted text (first 500 characters)
    st.write("Extracted Text Preview:")
    st.text_area("PDF Text (First 500 characters):", pdf_text[:500], height=200)

    # Input field for the user to ask a question
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Finding the answer..."):
            try:
                # Get the answer from the chatbot
                answer = chat_with_bot(pdf_text, question)
                st.write(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error querying the model: {e}")
