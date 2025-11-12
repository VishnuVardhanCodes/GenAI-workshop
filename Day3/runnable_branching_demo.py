from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

model = ChatOllama(model="tinyllama")

# Define prompts for branching
prompt_yes = PromptTemplate.from_template("Explain why {animal} can fly.")
prompt_no = PromptTemplate.from_template("Explain why {animal} cannot fly.")

# Simple branching logic: only birds and bats can fly (customize as needed)
def can_fly(input):
    return input["animal"].lower() in ["bird", "bat", "eagle", "parrot"]

branch_chain = RunnableBranch(
    (RunnableLambda(can_fly), prompt_yes | model | StrOutputParser()),
    prompt_no | model | StrOutputParser()
)

output = branch_chain.invoke({"animal": "elephant"})
print(output)
