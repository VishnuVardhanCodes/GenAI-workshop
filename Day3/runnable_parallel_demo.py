from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel

model = ChatOllama(model="tinyllama")

prompt_capital = PromptTemplate.from_template("What is the capital of {country}?")
prompt_population = PromptTemplate.from_template("What is the population of {country}?")

chain_capital = (
    prompt_capital
    | model
    | StrOutputParser()
)

chain_population = (
    prompt_population
    | model
    | StrOutputParser()
)

parallel_chain = RunnableParallel(
    capital=chain_capital,
    population=chain_population
)

output = parallel_chain.invoke({"country": "India"})
print(output)
