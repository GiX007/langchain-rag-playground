"""Create OpenAI embeddings (single and batch), rank documents by cosine similarity, and cache embeddings to avoid repeated API calls.

Run with: uv run src/langchain_rag_playground/embeddings/embedding_basics.py
"""

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load API keys from .env


# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embeddings() -> None:
    """Embed one text and print its size, first values and norm."""

    # Single text
    text = "What is an agent?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimension: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")  # ~1.0: OpenAI returns unit-length vectors, so dot product = cosine similarity


def batch_embeddings() -> None:
    """Embed several texts in one call and print size, first values and norm of each."""

    # Several texts, embedded in one API call
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embedding = embeddings.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")


def similarity_search() -> None:
    """Rank documents by cosine similarity to a query and print the scores."""

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # Embed docs and query
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    # Compute cosine similarities
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Return the cosine similarity of two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # Rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")


def embedding_caching() -> None:
    """Cache embeddings in a local store so the same text calls the API only once."""
    import tempfile

    from langchain_classic.embeddings.cache import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=store,
            namespace="exercise",
        )

        text = "What is Reinforcement Learning?"

        # First call - hits API
        print("First call (API):")
        vectors1 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors1)} documents")

        # Second call - from cache
        print("\nSecond call (Cache):")
        vectors2 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors2)} documents")

        # Verify same results
        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")


if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    # similarity_search()
    embedding_caching()  # See also the caching demo in production/cost_optimization.py