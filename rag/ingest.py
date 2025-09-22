# ingest_multimodal.py

import os
import glob
import hashlib
from typing import List

from langchain.schema import Document, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Unstructured local parsers
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text

# Ollama Multi-Modal LLM
from langchain_ollama import ChatOllama

DATA_PATH = "data/"
DB_PATH = "vectorstore/"

# Initialize Ollama (Bakllava for multimodal)
llm = ChatOllama(model="bakllava", temperature=0)
llm_text=ChatOllama(model="gemma:2b", temperature=0)

def summarize_element(el, file: str) -> str:
    """
    Summarize a single element (text, table, image) using Ollama bakllava.
    Always returns a string.
    """
    try:
        if el.category.lower() == "image":
            # Stub for images – currently not extracting raw image
            prompt = "Describe and summarize this image or chart in 2-3 sentences."
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt}
                # TODO: Add {"type": "image_url", "image_url": "file://path/to/extracted.png"}
            ])
            result = llm.invoke([msg])
            return str(result.content)
        else:
            # Handle text-like elements
            prompt = f"Summarize the following {el.category} in 2-3 sentences:\n{el.text}"
            result = llm_text.invoke(prompt)
            return str(result.content)
    except Exception as e:
        print(f"⚠️ Summarization failed for element in {os.path.basename(file)}: {e}")
        return ""


def load_and_summarize(limit: int = None) -> List[Document]:
    """
    Load documents, summarize each element using Ollama, return as Document objects.
    """
    files = glob.glob(os.path.join(DATA_PATH, "*"))
    if limit:
        files = files[:limit]

    summarized_docs = []

    for file in files:
        try:
            ext = os.path.splitext(file)[1].lower()

            # Parse documents locally
            if ext == ".pdf":
                elements = partition_pdf(file)
            elif ext == ".docx":
                elements = partition_docx(file)
            elif ext in [".txt", ".md"]:
                elements = partition_text(file)
            else:
                print(f"⚠️ Skipping unsupported file type: {file}")
                continue

            print(f"📂 Processing {os.path.basename(file)} with {len(elements)} elements...")

            for i, el in enumerate(elements, start=1):
                if (hasattr(el, "text") and el.text.strip()) or el.category.lower() == "image":
                    summary = summarize_element(el, file)
                    if not summary:
                        continue

                    # Stable doc_id
                    doc_id = f"{os.path.basename(file)}_{el.category}_{hashlib.sha256(str(el).encode('utf-8')).hexdigest()[:16]}"

                    summarized_docs.append(Document(
                        page_content=summary,
                        metadata={
                            "source": file,
                            "type": el.category,
                            "doc_id": doc_id
                        }
                    ))
                    print(f"   ✅ Summarized element {i}/{len(elements)} ({el.category})")

            print(f"✅ Finished {os.path.basename(file)} → {len(summarized_docs)} total summaries so far")

        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")

    return summarized_docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split long summaries into smaller chunks (~10 00 chars with 100 overlap).
    """
    if not docs:
        print("❌ No documents to split. Exiting.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100)

    chunks = splitter.split_documents(docs)

    print(f"✂️ Split {len(docs)} summaries into {len(chunks)} chunks")
    return chunks


def embed_and_store(docs: List[Document]):
    """
    Embed summaries and store them in Chroma vector database.
    """
    if not docs:
        print("❌ No chunks to embed. Exiting.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=DB_PATH)
    print(f"✅ Stored {len(docs)} summarized chunks in Chroma DB at {DB_PATH}")


if __name__ == "__main__":
    # 1. Load documents and summarize elements
    summaries = load_and_summarize(limit=10)

    # 2. Chunk long summaries
    chunks = chunk_documents(summaries)

    # 3. Embed and store in ChromaDB
    embed_and_store(chunks)
