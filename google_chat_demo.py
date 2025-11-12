import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Load environment variables
load_dotenv()

# Get your API key from .env
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Create a simple prompt
prompt = PromptTemplate.from_template(
    "You are an AI tutor. Answer clearly.\n\nQuestion: {question}"
)

# Build chain
chain = RunnableSequence(prompt | llm | StrOutputParser())

# Example usage
question = "Explain why self-consistency is useful for large language models."
output = chain.invoke({"question": question})

print("\nGemini says:\n")
print(output)

