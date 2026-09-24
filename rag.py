from importlib.resources import path
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_BASE_FILE = Path(__file__).resolve().parent / "knowledge"

def load_knowledge_base() -> list[Document]:
    """Read our stable fashion guide and turn them into LangChain documents with metadata."""
    documents: list[Document] = []
    for file_path in KNOWLEDGE_BASE_FILE.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={
            "source": path.name,
            "topic": path.stem,

            }))

        return documents


def initialize_rag() -> InMemoryVectorStore:
    """Building the Fashion knowledge base"""
    global _vector_store, _chunk_count

    if _vector_store is not None:
        return _vector_store

    documents = load_knowledge_base()

    print("Loading knowledge base...")


    #chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=550,
        chunk_overlap=80,
        length_function=len
    )

    chunks = splitter.split_documents(documents)
    _chunk_count = len(chunks)
    print(f"Created {_chunk_count}")



    #embedding
    print(f"Loading the embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    _vector_store = InMemoryVectorStore(embedding=embeddings)
    _vector_store.add_documents(chunks)

    print(f"Knowledge base loaded")
    return _vector_store


def search_fashion_knowledge_base(query: str, top_k: int = 3) -> list[Document]:
    """
        Search the knowledge base for relevant documents.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
        Returns:
            JSON containing the top_k most relevant and semantically related documents from the knowledge base.
    
    """


    vector_store = initialize_rag()
    results = vector_store.similarity_search(query, k=top_k)
    print(f" search_fashion_knowledge_base")
    print(f"Found {len(results)} relevant documents for query: '{query}'")

    payload = []

    for index, doc in enumerate(results, start=1):
        payload.append({
            "rank": index,
            "source": doc.metadata.get("source"),
            # "role": "assistant",
            "topic": doc.metadata.get("topic"),
            "content": doc.page_content,
            "metadata": doc.metadata
        }) 
    

    return json.dumps(payload, indent=2)


    