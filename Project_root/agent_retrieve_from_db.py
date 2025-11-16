from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

DB_FOLDER = "vector_db"

# Embedding model same as during storage
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing Chroma database
db = Chroma(
    persist_directory=DB_FOLDER,
    embedding_function=embeddings
)

# Retriever setup
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Model
llm = ChatOllama(model="llama3.2")   # or your model name

# RAG prompt template
template = """
You are an AI assistant. Use the following context from the PDFs to answer the question.
If the answer is not in the context, say "I don't know based on the PDFs."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

def ask_question(query):
    # Retrieve from vector DB
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build final prompt
    final_prompt = prompt.format(context=context, question=query)

    # Get model answer
    answer = llm.invoke(final_prompt)
    return answer.content


if __name__ == "__main__":
    print("🤖 PDF Retrieval Agent Ready!")
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:", ask_question(q))
