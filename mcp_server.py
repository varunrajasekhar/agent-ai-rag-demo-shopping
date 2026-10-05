from contextlib import redirect_stdout
import sys 
from mcp.server import MCPServer

from outfit_tools import plan_outfit_budget as _plan_outfit_budget
from tools import search_products as _search_products

mcp = MCPServer(
    "Outfit Shopping MCP",
    instructions=
    "provides outfit budget planning and external retailer product search"
)

@mcp.tool()
def plan_outfit_budget(total_budget: float) -> str:
    """split a total outfit budget across top, bottom and shoes."""
    with redirect_stdout(sys.stderr):
        return _plan_outfit_budget(total_budget)
    
@mcp.tool()
def search_products(
    stores: list[str],
    product_type: str,
    shopper_request: str,
    max_price: float | None = None,
) -> str:
    """Search grounded retailer products using the existing RAG pipeline."""
    with redirect_stdout(sys.stderr):
        return _search_products(
            stores=stores,
            product_type=product_type,
            shopper_request=shopper_request,
            max_price=max_price,
        )
if __name__ == "__main__":
    mcp.run(transport="stdio")