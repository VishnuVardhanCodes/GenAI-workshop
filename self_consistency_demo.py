# 4self_consistency.py
# Works with latest LangChain (2025 versions)
# pip install langchain langchain-core langchain-community ollama

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from collections import Counter

# ---- SETTINGS ----
MODEL_NAME = "tinyllama"
QUESTION = "If there are 3 red balls and 2 blue balls in a box, and you pick one randomly, what is the probability it's red?"
NUM_SAMPLES = 5

# ---- DEFINE MODEL ----
llm = Ollama(model=MODEL_NAME, temperature=0.9)

prompt = PromptTemplate.from_template(
    "Think step by step and answer carefully.\n"
    "Question: {question}\n\n"
    "Final Answer format: 'Answer: <value>'"
)

# Build chain (Prompt → LLM → Parser)
chain = RunnableSequence(prompt | llm | StrOutputParser())

answers = []
for i in range(NUM_SAMPLES):
    print(f"\n🧠 Run {i+1}/{NUM_SAMPLES}")
    output = chain.invoke({"question": QUESTION})
    print(output)
    if "Answer:" in output:
        ans = output.split("Answer:")[-1].strip().split()[0]
        answers.append(ans)

if answers:
    most_common = Counter(answers).most_common(1)[0][0]
    print("\n===============================")
    print("All extracted answers:", answers)
    print("✅ Final Self-Consistent Answer:", most_common)
else:
    print("No valid answers extracted.")