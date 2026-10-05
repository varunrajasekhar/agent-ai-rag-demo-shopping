# ShopSense Day 3: Diverse RAG Search + Verified Price

This version keeps the working Day 3 design and adds two focused improvements:

1. **Diversity-first search:** RAG provides three product-specific retailer
   terms. Firecrawl searches all three and the system keeps at most one unique
   product from each search.
2. **Hard price validation:** when the shopper supplies a maximum price, the
   adapter scrapes each candidate product page with Firecrawl's `product`
   format and accepts it only when an available current price is within budget.

## Example

For `professional, comfortable shirts`, RAG may produce:

```text
cotton poplin
linen blend
button-up
```

The tool then aims to return:

```text
1 product from cotton poplin
1 product from linen blend
1 product from button-up
```

If one of those searches finds nothing, one product-type-only fallback search
tries to fill the missing spaces.

## Complete flow

1. The Ollama agent extracts the retailer, literal product type, active
   department, complete shopper request, and optional maximum price.
2. `search_products` runs RAG in Python; the model cannot rewrite its terms.
3. Python performs three separate RAG-keyword searches.
4. Each search requests only six candidate results and contributes at most one
   unique product.
5. The adapter rejects non-US URLs, wrong categories, children's products,
   search pages, and duplicate URLs.
6. If a budget was supplied, each candidate page is scraped with
   `formats=["product"]`; missing, unavailable, foreign-currency, and
   over-budget prices are rejected.
7. The tool returns no more than three real products.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your Firecrawl key in `.env`:

```env
FIRECRAWL_API_KEY=fc-da5fafb2a06241a693214be3f07cd3ed
```

Make sure Ollama is running and install the model:

```bash
ollama pull qwen2.5:7b
```

Run:

```bash
python main.py
```

Try:

```text
Search Zara for professional, comfortable shirts under $100
```

## Terminal visibility

The terminal now shows:

- each RAG keyword and Firecrawl query;
- which of the three searches is running;
- how many valid products that search contributed;
- the total unique product count;
- whether a candidate's verified price was accepted or rejected.

## Cost control

- Every RAG keyword search uses `limit=6`, reduced from 12.
- Each keyword contributes at most one product.
- Product-page price scraping happens only when `max_price` is supplied.
- Scraping stops as soon as one qualifying product is found for that keyword.
- A fourth generic search runs only when the three RAG searches return fewer
  than three products.

## Teaching files

- `rag.py`: retrieval and product-specific keyword extraction.
- `knowledge/product_search_terms.md`: retailer vocabulary used by RAG.
- `tools.py`: three-search diversity strategy, deduplication, and fallback.
- `retailer_adapter.py`: Firecrawl search, category validation, US URL
  validation, structured product-price extraction, and budget filtering.
- `agent.py`: the small model prompt and single exposed product-search tool.
