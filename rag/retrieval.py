from langchain_community.vectorstores import chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever



# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Initialize vector store
vectordb = chroma.Chroma(persist_directory="db", embedding_function=embedding_model)
# Initialize LLMs
google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
ollama_llm = ChatOllama(model="llama2", temperature=0)
# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides concise and accurate answers based on the provided context."),
    ("user", "Use the following context to answer the question. If the context is insufficient, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}")
])
# Initialize multi-query retriever
base_retriever = MultiQueryRetriever.from_llms(
    llms=google_llm,
    vectorstore=vectordb,
    prompt=prompt_template,
    top_k=3
)

#Defining Contextual Compression retreiver
retriever = ContextualCompressionRetriever(
    base_compressor=ollama_llm,
    base_retriever=base_retriever
)
# Example usage
query = "What are the key insights from the documents?"
docs = retriever.invoke(query)

print(docs)