import anyio
import os
import sys
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool 
from mcp import Client, StdioServerParameters

PROJECT_DIR = Path(__file__).resolve().parent

def _server_parameters() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_DIR / "mcp_server.py")],
        cwd=str(PROJECT_DIR),
        env=dict(os.environ)
    )

async def _call_mcp_tool(name: str, arguments: dict) -> str:
    async with Client(_server_parameters()) as client:
        result = await client.call_tool(name, arguments)
        text_parts = [
            item.text for item in result.content
            if getattr(item,"type", None) == "text"
        ]
        if text_parts:
            return "\n".join(text_parts)
        
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return str(structured)
        
        return str(result)

def _call(name: str, arguments: dict) -> str:
    return anyio.run(_call_mcp_tool, name, arguments)

@tool
def mcp_plan_outfit_budget(total_budget: float) -> str:
    """Split a total outfit budget across top, bottom and shoes."""
    return _call("plan_outfit_budget", {"total_budget": total_budget})


@tool 
def mcp_search_products(
    stores: list[str],
    product_type: str,
    shopper_request: str,
    max_price: Optional[float] = None,
) -> str:
    """Search external retailer products using the existing RAG pipeline."""
    return _call(
        "search_products",
        {
            "stores": stores,
            "product_type": product_type,
            "shopper_request": shopper_request,
            "max_price": max_price,
        },
    )