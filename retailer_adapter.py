import re
from typing import Optional

from firecrawl import Firecrawl


class FirecrawlRetailerAdapter:
    """Adapt one generic retailer-search contract to Firecrawl."""

    SEARCH_RESULT_LIMIT = 10

    STORE_CONFIG = {
        "zara": {
            "name": "Zara", "domain": "zara.com",
            "product_path": "-p", "us_path": "/us/",
        },
        "h&m": {
            "name": "H&M", "domain": "hm.com",
            "product_path": "productpage", "us_path": "/en_us/",
        },
        "h and m": {
            "name": "H&M", "domain": "hm.com",
            "product_path": "productpage", "us_path": "/en_us/",
        },
        "hém": {
            "name": "H&M", "domain": "hm.com",
            "product_path": "productpage", "us_path": "/en_us/",
        },
        "mango": {
            "name": "Mango", "domain": "shop.mango.com",
            "product_path": "/p/", "us_path": "/us/",
        },
        "mongo": {
            "name": "Mango", "domain": "shop.mango.com",
            "product_path": "/p/", "us_path": "/us/",
        },
    }

    PRODUCT_IDENTIFIERS = {
        "shirt": ("shirt", "blouse", "overshirt"),
        "top": (
            "top",
            "blouse",
            "shirt",
            "polo",
            "tank",
            "camisole",
            "tunic",
            "tee",
            "t-shirt",
            "vest",
        ),
        "pants": (
            "pant",
            "pants",
            "trouser",
            "trousers",
            "slack",
            "slacks",
            "chino",
            "chinos",
            "jean",
            "jeans",
        ),
        "blazer": ("blazer", "jacket"),
        "dress": ("dress",),
        "skirt": ("skirt",),
        "shoes": (
            "shoe", "shoes", "loafer", "loafers", "moccasin",
            "sneaker", "sneakers", "flat", "flats", "pump", "pumps",
            "heel", "heels", "boot", "boots",
        ),
    }
    WRONG_CATEGORY_TERMS = {
        "shirt": ("dress", "skirt", "bikini", "swimsuit", "trouser", "pants"),
        "top": ("dress", "skirt", "bikini", "swimsuit", "trouser", "pants"),
        "pants": ("dress", "skirt", "shirt", "blouse", "bikini", "swimsuit"),
        "blazer": ("dress", "skirt", "shirt", "blouse", "bikini", "swimsuit"),
        "dress": ("skirt", "bikini", "swimsuit", "trouser", "pants"),
        "skirt": ("dress", "bikini", "swimsuit", "trouser", "pants"),
        "shoes": (
            "skirt", "shirt", "blouse", "trouser", "pants",
            "bag", "handbag", "belt",
        ),
    }
    CHILD_TERMS = (
        "baby", "child", "children", "kid", "kids",
        "girl", "girls", "boy", "boys",
    )

    def __init__(self, api_key: str):
        self.client = Firecrawl(api_key=api_key)

    @staticmethod
    def _print_rejected(
        title: str,
        url: str,
        reason: str,
    ) -> None:
        """Print why a Firecrawl candidate was rejected."""
        print(f"[Rejected] {title}")
        print(f"           Reason: {reason}")
        print(f"           URL: {url or '(missing URL)'}")
    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        return re.search(
            rf"(?<![a-z]){re.escape(term)}(?![a-z])", text
        ) is not None

    @staticmethod
    def _read_field(value, *field_names):
        """Read from either a dictionary or an SDK response object."""
        if value is None:
            return None
        for field_name in field_names:
            if isinstance(value, dict):
                if field_name in value:
                    return value[field_name]
            else:
                result = getattr(value, field_name, None)
                if result is not None:
                    return result
        return None

    def _matches_product_type(
        self, title: str, url: str, product_type: str
    ) -> bool:
        """Reject obvious wrong categories before results reach the model."""
        identity_text = f"{title} {url}".lower()

        if any(
            self._contains_term(identity_text, term)
            for term in self.CHILD_TERMS
        ):
            return False

        if any(
            self._contains_term(identity_text, term)
            for term in self.WRONG_CATEGORY_TERMS.get(product_type, ())
        ):
            return False

        expected_terms = self.PRODUCT_IDENTIFIERS.get(product_type)
        if expected_terms:
            return any(
                self._contains_term(identity_text, term)
                for term in expected_terms
            )

        return all(
            self._contains_term(identity_text, word)
            for word in product_type.split()
        )

    def _get_verified_product(
        self,
        url: str,
        max_price: float,
    ) -> Optional[dict]:
        """Return product details when an available variant is in budget."""
        result = self.client.scrape(url=url, formats=["product"])
        product = self._read_field(result, "product")
        if not product:
            return None

        eligible_prices = []
        for variant in self._read_field(product, "variants") or []:
            availability = self._read_field(variant, "availability")
            in_stock = self._read_field(
                availability, "inStock", "in_stock"
            )
            if in_stock is False:
                continue

            price_information = self._read_field(variant, "price")
            amount = self._read_field(price_information, "amount")
            if amount is None:
                continue

            try:
                amount = float(amount)
            except (TypeError, ValueError):
                continue

            if amount <= 0 or amount > max_price:
                continue

            currency = self._read_field(price_information, "currency")
            if currency and str(currency).upper() != "USD":
                continue

            formatted = self._read_field(price_information, "formatted")
            eligible_prices.append(
                {
                    "amount": amount,
                    "currency": currency or "USD",
                    "formatted": formatted or f"${amount:.2f}",
                }
            )

        if not eligible_prices:
            return None

        lowest_price = min(
            eligible_prices,
            key=lambda candidate: candidate["amount"],
        )
        return {
            "name": self._read_field(product, "title") or "Product",
            "description": self._read_field(product, "description") or "",
            "price": lowest_price,
        }

    def search(
        self,
        store: str,
        product_type: str,
        search_keyword: Optional[str],
        department: str,
        max_price: Optional[float] = None,
        max_results: int = 1,
        excluded_urls: Optional[set[str]] = None,
    ) -> dict:
        """Search a retailer and return category- and price-valid products."""
        store_key = str(store).strip().lower().replace("\\&", "&")
        config = self.STORE_CONFIG.get(store_key)
        if not config:
            raise ValueError(f"Unsupported retailer: {store}")

        normalized_product_type = " ".join(str(product_type).lower().split())
        if not normalized_product_type:
            raise ValueError("Product type cannot be empty")

        keyword = " ".join(str(search_keyword or "").split())
        normalized_department = str(department).strip().lower()
        if normalized_department not in {"women", "men"}:
            raise ValueError("Department must be 'women' or 'men'")

        price = float(max_price) if max_price not in (None, "") else None
        if price is not None and price <= 0:
            raise ValueError("Maximum price must be greater than zero")
        # Firecrawl already filters the retailer domain. Avoid site/inurl
        # operators here: they returned unrelated pages in actual searches.
        # Keep strict US/product URL checks on the returned candidates.
        # For shoes, a specific style is clearer than "shoes loafers".
        category_text = (
            "" if normalized_product_type == "shoes" and keyword
            and keyword.lower() in {"loafers", "flat", "sneakers", "boots"}
            else normalized_product_type
        )
        external_query = " ".join(part for part in (
            config["name"],
            normalized_department,
            category_text,
            keyword,
        ) if part)

        print(f"\n[Adapter] Retailer: {config['name']}")
        print(f"[Adapter] RAG keyword: {keyword or '(product type only)'}")
        print(f"[Adapter] Firecrawl query: {external_query}")

        response = self.client.search(
            query=external_query,
            include_domains=[config["domain"]],
            country="US",
            limit=self.SEARCH_RESULT_LIMIT,
            timeout=30000,
        )

        web_results = list(response.web or [])

        print(
            f"[Adapter] Firecrawl returned "
            f"{len(web_results)} candidate results"
        )

        products = []
        excluded_urls = excluded_urls or set()

        for item in web_results:
            url = item.url or ""
            title = item.title or "Product"
            normalized_url = url.lower()

            if url in excluded_urls:
                self._print_rejected(
                    title,
                    url,
                    "Duplicate product URL",
                )
                continue

            if config["domain"] not in normalized_url:
                self._print_rejected(
                    title,
                    url,
                    "URL is from the wrong retailer",
                )
                continue

            if config["product_path"].lower() not in normalized_url:
                self._print_rejected(
                    title,
                    url,
                    "URL is not a product-detail page",
                )
                continue

            if config["us_path"] not in normalized_url:
                self._print_rejected(
                    title,
                    url,
                    "URL is not from the US website",
                )
                continue

            if not self._matches_product_type(
                title=title,
                url=url,
                product_type=normalized_product_type,
            ):
                self._print_rejected(
                    title,
                    url,
                    (
                        "Product category or department does not "
                        f"match {normalized_product_type}"
                    ),
                )
                
                continue

            verified_product = None
            if price is not None:
                try:
                    verified_product = self._get_verified_product(
                        url=url,
                        max_price=price,
                    )
                except Exception as error:
                    print(f"[Price] Could not verify {url}: {error}")
                    continue

                if verified_product is None:
                    self._print_rejected(
                        title,
                        url,
                        (
                            "No verified available USD price "
                            f"at or below ${price:g}"
                        ),
                    )
                    continue

                print(
                    f"[Price] Accepted: "
                    f"{verified_product['price']['formatted']}"
                )

            normalized_product = {
                "name": (
                    verified_product["name"] if verified_product else title
                ),
                "store": config["name"],
                "url": url,
                "description": (
                    verified_product["description"]
                    if verified_product
                    else item.description or ""
                ),
                "matched_rag_keyword": keyword or None,
            }
            if verified_product:
                normalized_product["price"] = verified_product["price"]

            products.append(normalized_product)
            if len(products) >= max_results:
                break

        return {
            "store": config["name"],
            "firecrawl_query": external_query,
            "search_results_received": len(web_results),
            "products": products,
        }
