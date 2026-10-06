"""Compare vector, BM25 and two hybrid searches (LangChain's EnsembleRetriever and our own RRF) on exact-code and meaning-based queries.

Run with: uv run src/langchain_rag_playground/retrieval/hybrid_search_comparison.py
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load API keys from .env


# Documents with both semantic content AND specific identifiers
documents = [
    Document(page_content='Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.', metadata={'type': 'product'}),
    Document(page_content='For network connectivity issues, first check the ethernet cable and router status lights.', metadata={'type': 'troubleshooting'}),
    Document(page_content='Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.', metadata={'type': 'error'}),
    Document(page_content='The authentication process requires valid credentials. Use OAuth2 for secure API access.', metadata={'type': 'auth'}),
    Document(page_content='Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.', metadata={'type': 'config'}),
    Document(page_content='WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.', metadata={'type': 'compliance'}),
]

print(f"Loaded {len(documents)} documents.")


# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

ids = [f"doc-{i}" for i in range(len(documents))]
vector_store = Chroma.from_documents(
    documents,
    embeddings,
    ids=ids,
    collection_name='hybrid_test'
)

# Create vector retriever
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(f"Vector retriever: {vector_retriever}")

# Create BM25 retriever that works on raw text
bm25_retriever = BM25Retriever.from_documents(documents, k=3)
print(f"BM25 retriever: {bm25_retriever}")

# Combine with EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # Equal weight to both
)
print(f"Hybrid retriever (vector+bm25): {ensemble_retriever}")


# Hybrid retrieval using Reciprocal Rank Fusion (RRF). This is what EnsembleRetriever does internally
def hybrid_retrieve(query: str, retrievers: list[BaseRetriever], weights: list[float], k: int = 3, rrf_k: int = 60) -> list[Document]:
    """Combine multiple retrievers using weighted Reciprocal Rank Fusion."""
    doc_scores: dict[str, tuple[float, Document]] = {}  # Maps page_content -> (score, doc)

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content
            # RRF: top-ranked docs get higher scores; docs found by both retrievers add up their scores
            rrf_score = weight * (1.0 / (rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    # Sort by fused score, highest first, and keep the top k
    ranked = sorted(doc_scores.values(), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


# Wrap the function as a Runnable, so it has .invoke() like the other retrievers
hybrid_retriever = RunnableLambda(
    lambda q: hybrid_retrieve(q, [bm25_retriever, vector_retriever], [0.5, 0.5])
)
print(f"Hybrid retriever (rrf): {hybrid_retriever}")


def test_query(query: str, name: str, retriever: Runnable[str, list[Document]]) -> list[Document]:
    """Test a query and show results."""
    results = retriever.invoke(query)
    print(f'\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'  {i+1}. {preview}')
    return results


# Test queries designed to challenge vector search
test_queries = [
    'SKU-7742X specifications',  # Exact product code
    'E_CONN_REFUSED error',  # Error code
    'How do I authenticate?',  # Semantic question
    'WCAG compliance',  # Acronym
    'router configuration',  # General semantic
]


for query in test_queries:
    print('=' * 60)

    # Vector only
    test_query(query, 'VECTOR', vector_retriever)

    # BM25 only
    test_query(query, 'BM25', bm25_retriever)

    # Hybrid 1
    test_query(query, 'HYBRID 1', ensemble_retriever)

    # Hybrid 2
    test_query(query, 'HYBRID 2', hybrid_retriever)