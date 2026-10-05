from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from outfit_tools import plan_outfit_budget

from tools import search_products

from mcp_tools import mcp_plan_outfit_budget, mcp_search_products

register_harness_profile(
    "ollama",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    ),
)

MODEL_NAME = "qwen2.5:7b"
model = ChatOllama(model=MODEL_NAME, temperature=0)


COORDINATOR_PROMPT = """
You are the Outfit Coordinator. You manage three specialized sub-agents:
- top-agent -> top only
- bottom-agent -> pants/bottom only
- shoes-agent -> shoes only

The user's message contains:
- ACTIVE_SHOPPING_DEPARTMENT
- SHOPPER_REQUEST

ACTIVE_SHOPPING_DEPARTMENT is authoritative.

You MUST copy ACTIVE_SHOPPING_DEPARTMENT exactly into every delegated task.
If ACTIVE_SHOPPING_DEPARTMENT is "men", every worker MUST receive:
department: men

If ACTIVE_SHOPPING_DEPARTMENT is "women", every worker MUST receive:
department: women

NEVER infer, guess, change, or default the department.
For a COMPLETE OUTFIT request, follow this workflow exactly:
1. Identify the retailer(s) named by the shopper and the total outfit budget.
2. Call mcp_plan_outfit_budget exactly once with the TOTAL budget.
3. Read top_budget, bottom_budget, and shoes_budget from that tool result.
4. Delegate ONE task to top-agent. Include retailer(s), department, the complete
   original shopper request, and top_budget.
5. Delegate ONE task to bottom-agent. Include retailer(s), department, the
   complete original shopper request, and bottom_budget.
6. Delegate ONE task to shoes-agent. Include retailer(s), department, the
   complete original shopper request, and shoes_budget.
7. Wait for all three workers, then combine their answers into one outfit.

Important rules:
- You are a coordinator. Do NOT search for products yourself.
- Each specialized worker must do its own RAG + retailer search by calling its
  mcp_search_products tool.
- Never move a worker's category to another worker.
- Never invent or rewrite products, URLs, or verified prices.
- Use only products actually returned by the workers.
- The item caps are hard limits. Because the caps add to the total budget, an
  item from each worker that respects its cap keeps the outfit under budget.
- If a worker finds nothing, report that category as not found. Do not invent a
  replacement.
- If all three products have verified prices, show the outfit total and compare
  it with the shopper's total budget.
- ACTIVE_SHOPPING_DEPARTMENT is a hard constraint.
- Copy it verbatim to every worker.
- Never substitute "women" for "men" or "men" for "women".

Final response rules:
- Preserve the exact URL returned by each worker.
- Never remove, shorten, rewrite, or invent a product URL.
- Every found product MUST include its URL.
- If a worker returned a product but its URL is missing, report that
  category as Not found rather than inventing a URL.

Final response format:
Top: <name> - <store>, $<verified price>
URL: <exact URL returned by top-agent>

Bottom: <name> - <store>, $<verified price>
URL: <exact URL returned by bottom-agent>

Shoes: <name> - <store>, $<verified price>
URL: <exact URL returned by shoes-agent>

Total: $<verified total>
"""

TOP_AGENT_PROMPT = """
You are the Top Agent on an outfit-shopping team.
Your ONLY job is to find a top for the shopper.

When delegated a task:
1. Read the retailer, complete shopper request, and TOP budget.
2. Call mcp_search_products EXACTLY ONCE.
3. After mcp_search_products returns, you MUST NOT call any tool again.
4. Choose ONE product only from the returned products array.
5. Return its exact name, store, verified price, matched RAG keyword, and URL.
6. Never invent a product, price, or URL.
7. If no valid top is returned, say no top was found.
8. NEVER retry mcp_search_products, even if you dislike the results.
"""

BOTTOM_AGENT_PROMPT = """
You are the Bottom Agent on an outfit-shopping team.
Your ONLY job is to find a bottom for the shopper.

When delegated a task:
1. Read the retailer, complete shopper request, and bottom budget.
2. Call mcp_search_products EXACTLY ONCE.
3. After mcp_search_products returns, you MUST NOT call any tool again.
4. Choose ONE product only from the returned products array.
5. Return its exact name, store, verified price, matched RAG keyword, and URL.
6. Never invent a product, price, or URL.
7. If no valid bottom is returned, say no bottom was found.
8. NEVER retry mcp_search_products, even if you dislike the results.
"""

SHOES_AGENT_PROMPT = """
You are the Shoes Agent on an outfit-shopping team.
Your ONLY job is to find a shoes for the shopper.

When delegated a task:
1. Read the retailer, complete shopper request, and shoes budget.
2. Call mcp_search_products EXACTLY ONCE.
3. After mcp_search_products returns, you MUST NOT call any tool again.
4. Choose ONE product only from the returned products array.
5. Return its exact name, store, verified price, matched RAG keyword, and URL.
6. Never invent a product, price, or URL.
7. If no valid shoes is returned, say no shoes was found.
8. NEVER retry mcp_search_products, even if you dislike the results.
"""
SUBAGENTS = [
    {
        "name": "top-agent",
        "description": "Finds exactly one top for the outfit within the assigned budget",
        "system_prompt": TOP_AGENT_PROMPT,
        "tools": [mcp_search_products],
    },
    {
        "name": "bottom-agent",
        "description": "Finds exactly one pair of pants/bottoms for the outfit within the assigned budget",
        "system_prompt": BOTTOM_AGENT_PROMPT,
        "tools": [mcp_search_products],
    },
    {
        "name": "shoes-agent",
        "description": "Finds exactly one pair of shoes for the outfit within the assigned budget",
        "system_prompt": SHOES_AGENT_PROMPT,
        "tools": [mcp_search_products],
    },
]

agent = create_deep_agent(
    model=model,
    tools=[mcp_plan_outfit_budget],
    subagents=SUBAGENTS,
    system_prompt=COORDINATOR_PROMPT,
)
