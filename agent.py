from deepagents import (
    create_deep_agent, 
    GeneralPurposeSubagentProfile,
    HarnessProfile, 
    register_harness_profile)
from langchain_ollama import ChatOllama
from rag import search_fashion_knowledge_base
from tools import search_products
register_harness_profile("ollama", HarnessProfile(
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
))

MODEL_NAME = "llama3.2:3b"

Model = ChatOllama(model=MODEL_NAME, temperature=0)

SHOPPER_INSTRUCTIONS = """
    You are a personal shopping agent working with small mock data as a json.
    The user provides the current shopping department:
    ACTIVE_SHOPPING_DEPARTMENT: women
    or 
    ACTIVE_SHOPPING_DEPARTMENT: men

    Tool Rule:
    For every request that asks to find, search, compare, show, recommend or suggest products,
    you must call the 'search_products' Python tool.

    Do NOT:
    - print JSON pretending to call the tool
    - tell the user what parameters you would use
    - answer from memory
    Actaully invoke 'search_products' with the correct parameters and return the results to the user.
"""

agent = create_deep_agent(
    model=Model,
    tools = [search_products, search_fashion_knowledge_base],
    system_prompt=SHOPPER_INSTRUCTIONS,
)

