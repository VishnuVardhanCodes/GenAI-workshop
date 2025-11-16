from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
import glob

# Load all PDFs
pdf_files = glob.glob("*.pdf")
documents = []

for file in pdf_files:
    loader = PyPDFLoader(file)
    documents.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

# Create embeddings + Chroma DB
embedder = HuggingFaceEmbeddings()
chroma_db = Chroma.from_documents(texts, embedder, collection_name="books_db")

# Create retriever
retriever = chroma_db.as_retriever()

# Use TinyLlama
llm = OllamaLLM(model="tinyllama")

# Create a simple prompt
template = """Use the following context to answer the question:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Runnable chain (modern LangChain)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Run the chain
query = "Explain the basics of Artificial Intelligence."
response = rag_chain.invoke(query)

print("\n🧠 Answer:\n", response)
