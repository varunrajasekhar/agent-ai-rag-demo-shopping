import json 
from decimal import Decimal, ROUND_DOWN

OUTFIT_BUDGET_WEIGHTS = {
    "top": Decimal("0.35"),
    "bottom": Decimal("0.30"),
    "shoes":Decimal("0.35")
}

def plan_outfit_budget(total_budget: float) -> str:
    """Split one hard outfit budget across top, bottom and shoes.
    The first two allocations are rounded down to cents. Shoes recive the remainder, so the three camps always add back to the shopper's exact budget.
    Args:
        total_budget: max amount the shopper wants to spend on the outfit
    """

    budget = Decimal(str(total_budget)).quantize(Decimal("0.01"))

    if budget <=0:
        raise ValueError("Total budget must be freater than zero")
    
    top = (budget * OUTFIT_BUDGET_WEIGHTS["top"]).quantize(Decimal("0.01"), rounding= ROUND_DOWN)
    bottom = (budget * OUTFIT_BUDGET_WEIGHTS["bottom"]).quantize(Decimal("0.01"), rounding= ROUND_DOWN)
    shoes = budget - top - bottom

    plan = {
        "total_budget": float(budget),
        "top_budget": float(top),
        "bottom_budget": float(bottom),
        "shoes_budget": float(shoes)
    }

    print(f"\n Budget Plan: Top: ${top:.2f} | Bottom: ${bottom:.2f} | Shoes: ${shoes:.2f}")
    return json.dumps(plan, indent=2)