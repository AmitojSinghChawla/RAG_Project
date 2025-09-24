# -------------------- Imports --------------------
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
load_dotenv(verbose=True)
from typing import List
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnableLambda

# -------------------- Configuration --------------------
DB_PATH = "vectorstore/"

# Existing vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

# -------------------- LLMs --------------------
# Summarizer for retrieved text
summarizer_llm = ChatOllama(model="gemma:2b", temperature=0)
# Final answer LLM
final_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


# -------------------- Multi-Query Retriever --------------------
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


output_parser = LineListOutputParser()

prompt =PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
)

llm_chain= prompt| summarizer_llm | output_parser

multi_retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    llm_chain=llm_chain
)


# -------------------- Contextual Compression (summarize retrieved docs) --------------------
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the document in 3-4 sentences, focusing on key insights."),
    ("user", "{doc}")
])
compressor = LLMChainExtractor.from_llm(summarizer_llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_retriever
)



def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


pretty_print_docs(docs)