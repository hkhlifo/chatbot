import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda

# Load environment variables (for API key etc.)
load_dotenv()

# Initialize app
app = FastAPI()

# LLM
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

# Load vector store from default insurance PDFs
def load_vectorstore():
    loader = DirectoryLoader(path='data_bajaj', glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_split, embeddings)
    return vector_store

vector_store = load_vectorstore()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Format inputs for RAG chain
def format_inputs(inputs: dict):
    docs = retriever.invoke(inputs["question"])
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    return {"question": combined_text}

# Build the RAG chain
chain = (
    RunnableLambda(format_inputs) |
    prompt_template |
    llm |
    parser
)

# Request model
class Query(BaseModel):
    question: str

# Webhook endpoint
@app.post("/webhook")
async def webhook(query: Query):
    response = chain.invoke({"question": query.question})
    return {"answer": response}
