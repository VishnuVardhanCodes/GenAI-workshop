from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence

model = ChatOllama(model="tinyllama")

prompt1 = PromptTemplate.from_template("Summarize in one sentence: {text}")
prompt2 = PromptTemplate.from_template("Now explain this in simple words: {summary}")

chain = (
    prompt1
    | model
    | StrOutputParser()
    | (lambda summary: prompt2.format(summary=summary))
    | model
    | StrOutputParser()
)

output = chain.invoke({"text": "LangChain enables sophisticated LLM application orchestration."})
print(output)
