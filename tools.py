import json
import os
import sys
from typing import Optional

from rag import build_fashion_search_plan
from retailer_adapter import FirecrawlRetailerAdapter


TARGET_PRODUCT_COUNT = 3
RAG_KEYWORDS_TO_SEARCH = 2
MAX_PRODUCTS_PER_SEARCH = 3
CANDIDATE_POOL_SIZE = 5



def _normalize_stores(stores) -> list[str]:
    """Accept either a Python list or a JSON list produced by a local model."""
    if isinstance(stores, str):
        try:
            parsed = json.loads(stores)
            stores = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            stores = [stores]
    if not isinstance(stores, list):
        stores = [stores]
    return [str(store) for store in stores]


def search_products(
    stores: list,
    product_type: str,
    shopper_request: str,
    max_price: Optional[float] = None,
) -> str:
    """Search two RAG angles and select three balanced products.
    Args:
        stores: Exactly the retailers requested by the shopper.
        product_type: The literal requested category, such as shirt or pants.
        shopper_request: The shopper's complete original request for RAG.
        department: The active department, either "women" or "men".
        max_price: Optional hard maximum price in US dollars.
    """


    stores = _normalize_stores(stores)

    # Python owns this handoff, so the LLM cannot rewrite the RAG keywords.
    rag_plan = build_fashion_search_plan(shopper_request, product_type)
    normalized_product_type = rag_plan["product_type"]
    rag_keywords = rag_plan["search_keywords"][:RAG_KEYWORDS_TO_SEARCH]
    department = os.getenv("ACTIVE_SHOPPING_DEPARTMENT","").strip().lower()
    print(f'selected department {department}')
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return json.dumps(
            {
                "product_type": normalized_product_type,
                "rag_search_keywords": rag_keywords,
                "search_strategy": (
                    "two RAG searches, five balanced candidates, "
                    "three final products"
                ),
                "adapter_queries": [],
                "products": [],
                "errors": ["FIRECRAWL_API_KEY is missing from .env"],
            },
            indent=2,
        )

    adapter = FirecrawlRetailerAdapter(api_key)
    products = []
    adapter_queries = []
    errors = []
    seen_urls = set()

    # Each list will contain the valid products returned
    # by one RAG-keyword search.
    candidate_groups = []

    for search_number, keyword in enumerate(
        rag_keywords,
        start=1,
    ):
        search_products_found = []

        for store in stores:
            try:
                remaining_for_search = (
                    MAX_PRODUCTS_PER_SEARCH
                    - len(search_products_found)
                )

                if remaining_for_search <= 0:
                    break

                result = adapter.search(
                    store=store,
                    product_type=normalized_product_type,
                    search_keyword=keyword,
                    department=department,
                    max_price=max_price,
                    max_results=remaining_for_search,
                    excluded_urls=seen_urls,
                )

                adapter_queries.append(
                    {
                        "search_number": search_number,
                        "store": result["store"],
                        "rag_keyword": keyword,
                        "query": result["firecrawl_query"],
                        "search_results_received": (
                            result["search_results_received"]
                        ),
                        "valid_products_found": len(
                            result["products"]
                        ),
                    }
                )

                for product in result["products"]:
                    if product["url"] in seen_urls:
                        continue

                    seen_urls.add(product["url"])
                    search_products_found.append(product)

                    if (
                        len(search_products_found)
                        >= MAX_PRODUCTS_PER_SEARCH
                    ):
                        break

            except Exception as error:
                errors.append(
                    f"{store} / {keyword}: {error}"
                )

        candidate_groups.append(search_products_found)

        print(
            f"[Sequential Search] Search "
            f"{search_number}/{RAG_KEYWORDS_TO_SEARCH} "
            f"('{keyword}') returned "
            f"{len(search_products_found)} valid products"
            , file=sys.stderr
        )


    # Alternate between the two searches so the first
    # search does not dominate the recommendations.
    candidate_pool = []

    largest_group_size = max(
        (len(group) for group in candidate_groups),
        default=0,
    )

    for product_index in range(largest_group_size):
        for group in candidate_groups:
            if product_index >= len(group):
                continue

            candidate_pool.append(
                group[product_index]
            )

            if len(candidate_pool) >= CANDIDATE_POOL_SIZE:
                break

        if len(candidate_pool) >= CANDIDATE_POOL_SIZE:
            break


    products = candidate_pool[:TARGET_PRODUCT_COUNT]

    print(
        f"[Selection] Candidate pool: "
        f"{len(candidate_pool)}/{CANDIDATE_POOL_SIZE}"
        , file=sys.stderr
    )

    print(
        f"[Selection] Final products: "
        f"{len(products)}/{TARGET_PRODUCT_COUNT}"
        , file=sys.stderr
    )
    return json.dumps(
        {
            "product_type": normalized_product_type,
            "rag_search_keywords": rag_keywords,
            "search_strategy": (
                "two RAG searches, five balanced candidates, "
                "three final products"
            ),
            "adapter_queries": adapter_queries,
            "products": products[:TARGET_PRODUCT_COUNT],
            "candidate_pool": candidate_pool,
            "candidate_pool_size": len(candidate_pool),
            "errors": errors,
        },
        indent=2,
    )
