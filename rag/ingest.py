import os
import glob
import hashlib
from typing import List

from langchain.schema import Document, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text

from langchain_ollama import ChatOllama

# -------------------- Configuration --------------------
DATA_PATH = "D:/Projects/RAG/data"
DB_PATH = "vectorstore/"

llm = ChatOllama(model="bakllava", temperature=0)  # multimodal for images

# -------------------- Helpers --------------------
def summarize_image(el, file: str) -> str:
    """
    Summarize only images/charts using the LLM.
    """
    try:
        prompt = "Describe and summarize this image or chart in 2-3 sentences."
        msg = HumanMessage(content=[{"type": "text", "text": prompt}])
        result = llm.invoke([msg])
        return str(result.content)
    except Exception as e:
        print(f"⚠️ Image summarization failed in {os.path.basename(file)}: {e}")
        return ""


# -------------------- Main Processing --------------------
def load_documents(limit: int = None) -> List[Document]:
    files = glob.glob(os.path.join(DATA_PATH, "*"))
    if limit:
        files = files[:limit]

    docs = []

    for file in files:
        try:
            ext = os.path.splitext(file)[1].lower()
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
                if el.category.lower() == "image":
                    content = summarize_image(el, file)
                else:
                    content = el.text or ""

                if not content.strip():
                    continue

                doc_id = f"{os.path.basename(file)}_{el.category}_{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"
                docs.append(Document(
                    page_content=content,
                    metadata={"source": file, "type": el.category, "doc_id": doc_id}
                ))

                print(f"   ✅ Processed element {i}/{len(elements)} ({el.category})")

            print(f"✅ Finished {os.path.basename(file)} → {len(docs)} total so far")

        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    if not docs:
        print("❌ No documents to split. Exiting.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks


def embed_and_store(docs: List[Document]):
    if not docs:
        print("❌ No chunks to embed. Exiting.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=DB_PATH)
    print(f"✅ Stored {len(docs)} chunks in Chroma DB at {DB_PATH}")
    print(len(vectordb._collection.get(include=["metadatas"])["metadatas"]))


# -------------------- Main --------------------
if __name__ == "__main__":
    docs = load_documents(limit=10)       # Step 1: Load + summarize only images
    chunks = chunk_documents(docs)        # Step 2: Chunk text
    embed_and_store(chunks)               # Step 3: Embed + store

