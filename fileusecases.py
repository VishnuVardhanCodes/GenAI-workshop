from utils import get_google_genai_model
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load Gemini model via util
llm = get_google_genai_model("gemini-pro")

# Prompt template for text summarization
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following meeting transcript in 5 bullet points:\n\n{text}"
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
summary = chain.invoke({"text": "The meeting discussed project deadlines and client feedback..."})
print(summary['text'])
