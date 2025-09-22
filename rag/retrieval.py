from langchain_community.vectorstores import chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers

from rag.ingest import DB_PATH

DB_PATH.mkdir(parents=True, exist_ok=True)

def get_retreiver():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = chroma.Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectordb.as_retriever( search_kwargs={"k": 3})


llm=ChatOllama(model=)