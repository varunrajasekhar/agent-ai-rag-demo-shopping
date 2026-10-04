import json 
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent /"data" / "stores.json"
store_aliases = {
    "HéM": "H&M",
    "H and M": "H&M",
    "Zara": "Zara",
    "Mango": "Mango",
}

def load_catalog():
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def search_products(
        stores: list[str] | None = None,
        query: str | None = None,
        department: str | None = None,
        max_price: float | None = None) -> str:
    """Read or filter products from the local mock catalog.

    Omit any filter to include all matching catalog data. With no department,
    return both men and women products, grouped by department and retailer.

    Args:
        stores: Retailer names to include; omit to include every retailer.
        query: Optional text to match against product names, descriptions, and tags.
        department: Optional "men" or "women" filter; omit for both departments.
        max_price: Optional maximum product price.

    Returns:
        JSON with unchanged catalog product records grouped by department and retailer.
    """

    catalog = load_catalog()
    canonical_stores = {store.casefold(): store for store in catalog}
    aliases = {alias.casefold(): canonical for alias, canonical in store_aliases.items()}
    requested_stores = stores if stores else list(catalog)
    selected_stores = []
    unsupported_stores = []
    for requested_store in requested_stores:
        canonical = aliases.get(requested_store.casefold())
        canonical = canonical or canonical_stores.get(requested_store.casefold())
        if canonical is None:
            unsupported_stores.append(requested_store)
        elif canonical not in selected_stores:
            selected_stores.append(canonical)

    requested_department = department.casefold() if department else None
    if requested_department in {"all", "both"}:
        requested_department = None
    departments = ("men", "women")
    query_words = query.casefold().split() if query and query.strip() else []
    results = {name: {} for name in departments}

    for store in selected_stores:
        for department_name in departments:
            if requested_department and requested_department != department_name:
                continue
            matches = []
            for product in catalog[store]:
                if product["department"].casefold() != department_name:
                    continue
                if max_price is not None and product["price"] > max_price:
                    continue
                searchable_text = " ".join((
                    product["name"],
                    product["description"],
                    " ".join(product["tags"]),
                )).casefold()
                if query_words and not any(word in searchable_text for word in query_words):
                    continue
                matches.append(product)
            results[department_name][store] = matches

    return json.dumps({
        "products": results,
        "unsupported_stores": unsupported_stores,
        "available_stores": list(catalog),
    }, indent=2, ensure_ascii=False)