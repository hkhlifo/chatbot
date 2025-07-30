import os
import shutil
import streamlit as st
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableMap
import atexit

# Load environment variables
load_dotenv()

api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro')


# Prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert insurance analyst AI. You are given:
1. A user query describing an insurance case in plain English.
2. A set of relevant document snippets or clauses from policy documents, contracts, or emails.

Your task:
- Extract relevant information
- Identify coverage or exclusions
- Explain decision in clear English
- Return response in structured format:
insurance covered, what are the conditions, matched_clauses, and finally the explanation

Query:
{question}
"""
)

parser = StrOutputParser()

# Load documents and create vector store only once (default data)
@st.cache_resource
def load_vectorstore():
    loader = DirectoryLoader(path='data_bajaj', glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_split, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Create vectorstore from uploaded PDFs
def create_vectorstore_from_pdfs(files):
    temp_dir = "uploaded_data"
    os.makedirs(temp_dir, exist_ok=True)
    pdf_paths = []

    for uploaded_file in files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        pdf_paths.append(file_path)

    loaders = [PyPDFLoader(path) for path in pdf_paths]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs_split, embeddings)

# Cleanup uploaded files after session ends
@atexit.register
def cleanup_uploaded_data():
    if os.path.exists("uploaded_data"):
        shutil.rmtree("uploaded_data")

# Streamlit App UI
st.set_page_config(page_title="Insurance Analyst AI", layout="centered")
st.title("📄 Insurance Analyst AI")
st.markdown("Ask insurance-related questions based on uploaded policy documents or defaults.")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload your own insurance PDFs (optional):", type=["pdf"], accept_multiple_files=True
)

# Use voice assistant
use_voice = st.checkbox("🎙️ Use voice input/output")

# Load appropriate vectorstore
if uploaded_files:
    vector_store = create_vectorstore_from_pdfs(uploaded_files)
else:
    vector_store = load_vectorstore()

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Format inputs for chain
def format_inputs(inputs: dict):
    docs = retriever.invoke(inputs["question"])
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    return {"question": combined_text}

# Create the final chain
chain = (
    RunnableLambda(format_inputs) |
    prompt_template |
    llm |
    parser
)

# Voice recognizer and speaker
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def listen():
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    return ""

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Get input
if use_voice:
    if st.button("Start Voice Query"):
        user_input = listen()
    else:
        user_input = ""
else:
    user_input = st.text_input("Ask your question:")

# Process and respond
if user_input:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})
        st.markdown("🧠 AI Response")
        st.write(response)
        if use_voice:
            speak(response)
