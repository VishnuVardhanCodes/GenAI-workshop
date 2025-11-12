from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# ---- CONFIG ----
MODEL_NAME = "tinyllama"  # or mistral, llama3, etc.
TEMPERATURE = 0.0  # deterministic for consistency

# ---- DEFINE MODEL ----
chat_model = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

# ---- DEFINE A CHAT PROMPT ----
# We'll use a "System" role to set behavior, then simulate conversation messages
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful and precise math tutor. Always explain step-by-step."),
    HumanMessage(content="Hi, can you help me solve a simple math problem?"),
    AIMessage(content="Of course! What problem would you like me to solve?"),
    HumanMessage(content="If I have 12 apples and eat 3, how many are left?"),
])

# ---- CHAIN SETUP ----
chain = RunnableSequence(prompt | chat_model | StrOutputParser())

# ---- RUN ----
output = chain.invoke({})
print("🤖 TinyLlama says:\n")
print(output)