from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from typing import List

MODEL_NAME = "tinyllama"  # swap if you have a stronger model
QUESTION = "A train travels 60 km in 1 hour and 30 minutes. What is its average speed in km/h?"
NUM_PROMPTS = 5

# deterministic model settings (no randomness)
llm = Ollama(model=MODEL_NAME, temperature=0.0, top_p=0.0, top_k=1)

# Several prompt templates (different styles / many prompts)
prompt_texts: List[str] = [
    # 1. direct question, no CoT hint
    "Question: {question}\nAnswer succinctly.",

    # 2. ask for steps first (explicit chain-of-thought instruction)
    "Question: {question}\nExplain your reasoning step-by-step, then provide the final answer on a single line starting with 'Final:'.",

    # 3. few-shot CoT (one example) + target
    (
        "You are a careful solver.\n\n"
        "Example:\n"
        "Q: If a car covers 120 km in 2 hours, what's its speed?\n"
        "Step 1: compute total hours -> 2\n"
        "Step 2: compute speed = 120 / 2 = 60\n"
        "Final: 60 km/h\n\n"
        "Now solve:\nQuestion: {question}\nShow steps then 'Final:'."
    ),

    # 4. 'think step by step' short prompt (explicit CoT trigger)
    "Think step by step and solve: {question}\nProvide step-by-step reasoning, then final answer.",

    # 5. extremely explicit structured CoT request
    (
        "You MUST show: (1) unit conversions, (2) numeric calculation, (3) final result line.\n"
        "Question: {question}\nStart with 'Step 1:' and end with 'Result: <value>'"
    ),
]

# Build runnable chains for each prompt (PromptTemplate -> LLM -> string output)
chains = []
for i, pt in enumerate(prompt_texts, start=1):
    tmpl = PromptTemplate.from_template(pt)
    chain = RunnableSequence(tmpl | llm | StrOutputParser())
    chains.append((f"Prompt-{i}", chain))


def run_all():
    print("=== Deterministic Chain-of-Thought Demo ===\n")
    for name, chain in chains:
        print(f"--- {name} ---")
        out = chain.invoke({"question": QUESTION})
        print(out.strip())
        print()


if __name__ == "__main__":
    run_all()