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
        stores: list[str], 
        query: str,
        department: str, 
        max_price: float | None = None) -> str:
    """Search one or more retailers in the local mock json file
    Args:
        stores (list): List of store names to search in the mock data.
        query (str): Search query string.
        department (str): Department to filter by (e.g., "women" or "men").
        max_price (float): Maximum price to filter by.
    Returns:
        JSON containing matching mock products from the requested stores.
    """

    catalog = load_catalog()
    results = []
    query_words = query.lower().split()
    stores = [store_aliases.get(store, store) for store in stores]
    for store in stores: 
        for product in catalog[store]:
            if product["department"].lower() != department.lower():
                continue
            if max_price is not None and product["price"] > max_price:
                continue
            searchable_text = (product["name"] + " " + product["description"]).lower()+" "+" ".join(product['tags']).lower()
            score = sum(1 for word in query_words if word in searchable_text)
            if score > 0:
                product['keyword_score'] = score
                results.append(product)
    results.sort(key=lambda x: x['keyword_score'])
    return json.dumps(results, indent=2)