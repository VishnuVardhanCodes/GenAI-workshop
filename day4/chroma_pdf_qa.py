from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# 1️⃣ Load PDF
pdf_path = "yourfile.pdf"  # change this to your PDF name
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2️⃣ Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3️⃣ Create Embeddings and Store in Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embedding=embeddings, persist_directory="./chroma_db")

# 4️⃣ Define the LLM (TinyLlama from Ollama)
llm = Ollama(model="tinyllama")

# 5️⃣ Create RetrievalQA Chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 6️⃣ Ask Questions
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print("\nAnswer:", answer)
