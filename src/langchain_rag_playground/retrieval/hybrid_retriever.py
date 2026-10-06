"""Hybrid retriever: combine BM25 (keywords) and vector search (meaning) with weighted Reciprocal Rank Fusion (RRF).

Run with: uv run src/langchain_rag_playground/retrieval/hybrid_retriever.py
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load API keys from .env


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


class HybridRetriever:
    """Hybrid retriever that keeps a BM25 index and a Chroma vector store in sync."""

    def __init__(self, documents: list[Document], bm25_weight: float = 0.5, k: int = 4) -> None:
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create vector store and retriever
        self.vectorstore = Chroma.from_documents(documents, self.embeddings, collection_name="hybrid_search")
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    def search(self, query: str) -> list[Document]:
        """Run hybrid search using weighted RRF."""
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Add new documents to both retrievers."""
        # Add to vector store
        self.vectorstore.add_documents(documents)

        # Recreate BM25 from all stored docs (it doesn't support incremental adds)
        all_docs = self.vectorstore.get()
        all_documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        self.bm25_retriever = BM25Retriever.from_documents(all_documents, k=self.k)


if __name__ == "__main__":

    # Usage example (calls the OpenAI embeddings API)
    documents = [
        Document(page_content='Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.', metadata={'type': 'product'}),
        Document(page_content='For network connectivity issues, first check the ethernet cable and router status lights.', metadata={'type': 'troubleshooting'}),
        Document(page_content='Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.', metadata={'type': 'error'}),
        Document(page_content='The authentication process requires valid credentials. Use OAuth2 for secure API access.', metadata={'type': 'auth'}),
        Document(page_content='Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.', metadata={'type': 'config'}),
        Document(page_content='WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.', metadata={'type': 'compliance'}),
    ]

    retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
    results = retriever.search("SKU-7742X specifications")

    for doc in results:
        print(doc.page_content[:100])