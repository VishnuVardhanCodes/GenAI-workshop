import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Folder paths
PDF_FOLDER = "pdfs"
DB_FOLDER = "vector_db"

# Use HuggingFace all-MiniLM model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdfs():
    docs = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            docs.extend(loader.load())
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

def store_to_chroma(chunks):
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER
    )
    print("✅ PDF content stored successfully in Chroma DB!")

if __name__ == "__main__":
    print("📂 Loading PDFs...")
    documents = load_pdfs()

    print("✂️ Splitting documents...")
    chunks = split_docs(documents)

    print("💾 Storing chunks in Chroma DB...")
    store_to_chroma(chunks)
