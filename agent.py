from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain_ollama import ChatOllama

from tools import search_products


register_harness_profile(
    "ollama",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    ),
)

MODEL_NAME = "qwen2.5:7b"
model = ChatOllama(model=MODEL_NAME, temperature=0)

SHOPPER_INSTRUCTIONS = """
You are a personal shopping assistant.

The user's message contains ACTIVE_SHOPPING_DEPARTMENT: women or men.

For a product-shopping request, call search_products exactly once. Pass:
- stores: exactly the retailer or retailers named by the shopper
- product_type: the literal category, such as shirt, pants, or blazer
- shopper_request: the shopper's complete original request
- department: the active shopping department
- max_price: the maximum price supplied by the shopper, if any

search_products internally performs:
RAG retrieval -> two product-specific search keywords -> retailer adapter ->
two Firecrawl searches -> US URL and product-category validation ->
optional product-page price verification -> up to five balanced candidates ->
three final products.

Rules:
- Never create or rewrite the RAG keywords yourself.
- Never change the requested product type or retailer.
- Do not print or simulate a tool call. Actually call the tool and wait.
- Recommend only products returned in the products array.
- Copy every URL exactly. Never invent or rewrite products, URLs, or prices.
- If max_price is supplied, it is a hard constraint. Every returned product
  has a verified current price at or below that amount.
- If fewer than three products return, show only those products.
- Never substitute dresses, skirts, children's items, search-result pages, or
  products from another retailer.

For each product show its name, retailer, verified price when returned, matched
RAG keyword, why it may match using only returned information, and direct URL.
"""

agent = create_deep_agent(
    model=model,
    tools=[search_products],
    system_prompt=SHOPPER_INSTRUCTIONS,
)
