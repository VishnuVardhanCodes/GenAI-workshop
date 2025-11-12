from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# 1️⃣ Load the PDF
loader = PyPDFLoader(r"C:\Users\POLLA VISHNU VARDHAN\AIWorkshop\college_landing_page.pdf")
pages = loader.load()

# 2️⃣ Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

# 3️⃣ Create vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)

# 4️⃣ Initialize the LLM
llm = OllamaLLM(model="tinyllama")

# 5️⃣ Ask questions in a loop
print("✅ PDF loaded and ready. Ask your question below (type 'exit' to quit):\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    context = " ".join([d.page_content for d in docs])
    prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    print(f"\n🤖 {response}\n")

#python ask_pdf.py (running the pdf) to ask the questions for it !
