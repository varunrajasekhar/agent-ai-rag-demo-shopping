import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"


SEARCH_TERMS_BY_PRODUCT = {
    "shirt": (
        "cotton poplin",
        "linen blend",
        "button-up",
        "collared",
        "poplin",
        "linen",
        "cotton",
        "viscose",
        "rayon",
        "breathable",
        "lightweight",
        "wrinkle resistant",
        "relaxed fit",
        "regular fit",
        "tailored",
    ),
    "top": (
        "collared",
        "cotton poplin",
        "linen blend",
        "linen",
        "cotton",
        "viscose",
        "rayon",
        "breathable",
        "lightweight",
        "relaxed fit",
        "regular fit",
    ),
    "pants": (
        "wide-leg",
        "straight fit",
        "relaxed fit",
        "tailored",
        "linen blend",
        "linen",
        "cotton",
        "viscose",
        "breathable",
        "lightweight",
        "stretch",
    ),
    "blazer": (
        "unstructured",
        "tailored",
        "linen blend",
        "linen",
        "cotton",
        "lightweight",
        "breathable",
        "stretch",
        "wrinkle resistant",
    ),
    "dress": (
        "shirt dress",
        "midi dress",
        "linen blend",
        "linen",
        "cotton",
        "viscose",
        "rayon",
        "breathable",
        "lightweight",
        "relaxed fit",
    ),
    "skirt": (
        "midi skirt",
        "tailored",
        "linen blend",
        "linen",
        "cotton",
        "viscose",
        "lightweight",
        "stretch",
    ),
    "shoes": (
        "loafers",
        "dress shoes",
        "flat shoes",
        "low heel",
        "block heel",
        "sneakers",
        "leather shoes",
        "comfortable",
    ),
}

PRODUCT_TYPE_ALIASES = {
    "shirts": "shirt",
    "blouse": "shirt",
    "blouses": "shirt",
    "tops": "top",
    "trousers": "pants",
    "trouser": "pants",
    "pant": "pants",
    "blazers": "blazer",
    "jackets": "blazer",
    "jacket": "blazer",
    "dresses": "dress",
    "skirts": "skirt",
    "shoe": "shoes",
    "footwear": "shoes",
    "loafer": "shoes",
    "loafers": "shoes",
}

_vector_store: InMemoryVectorStore | None = None
_chunk_count = 0

# Standardizes product categories for consistent keyword lookup
def normalize_product_type(product_type: str) -> str:
    """Normalize common singular/plural product names."""
    normalized = " ".join(str(product_type).lower().split())
    if normalized in SEARCH_TERMS_BY_PRODUCT:
        return normalized
    if normalized in PRODUCT_TYPE_ALIASES:
        return PRODUCT_TYPE_ALIASES[normalized]

def load_knowledge_documents() -> list[Document]:
    """Read stable fashion guides and turn them into LangChain Documents."""
    documents: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name, "topic": path.stem},
            )
        )
    return documents


def initialize_rag() -> InMemoryVectorStore:
    """Build the fashion knowledge base once."""
    global _vector_store, _chunk_count
    if _vector_store is not None:
        return _vector_store

    print("\nLoading fashion knowledge documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=80)
    chunks = splitter.split_documents(load_knowledge_documents())
    _chunk_count = len(chunks)
    print(f"Created {_chunk_count} knowledge chunks")

    print("Loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    _vector_store = InMemoryVectorStore(embedding=embeddings)
    _vector_store.add_documents(chunks)
    print("Fashion knowledge base is ready")
    return _vector_store


def _extract_search_keywords(
    documents: list[Document],
    product_type: str,
    limit: int = 5,
) -> list[str]:
    """Extract only RAG terms that apply to the requested product type."""
    allowed_terms = SEARCH_TERMS_BY_PRODUCT.get(
        normalize_product_type(product_type), ()
    )
    print(f"allowed terms {allowed_terms}")
    retrieved_text = "\n".join(document.page_content.lower() for document in documents)


    print(f"retrieved text: {retrieved_text}")
    
    keywords: list[str] = []
    for term in allowed_terms:
        if term not in retrieved_text:
            continue
        if any(term in existing or existing in term for existing in keywords):
            continue
        keywords.append(term)
        if len(keywords) == limit:
            break
    return keywords


def build_fashion_search_plan(
    query: str,
    product_type: str,
    k: int = 4,
    keyword_limit: int = 3,
) -> dict:
    """Retrieve RAG chunks and build a product-specific keyword array."""
    documents = initialize_rag().similarity_search(query, k=k)
    print(f"document retrieved by Rag: {documents}")
    normalized_type = normalize_product_type(product_type)
    keywords = _extract_search_keywords(
        documents, product_type=normalized_type, limit=keyword_limit
    )
    # print(f'extracted keywords {keywords}')
    results = [
        {
            "rank": index,
            "source": document.metadata.get("source"),
            "topic": document.metadata.get("topic"),
            "content": document.page_content,
        }
        for index, document in enumerate(documents, start=1)
    ]
    # print(f"final results: {results}")
    return {
        "product_type": normalized_type,
        "search_keywords": keywords,
        "results": results,
    }
