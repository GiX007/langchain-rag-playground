"""Store documents in Chroma (raw client and LangChain) and search them: plain search, search with scores, and metadata filtering.

Run with: uv run src/langchain_rag_playground/vector_stores/chroma_basics.py
"""

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load API keys from .env


# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def print_embeddings_info() -> None:
    """Print information about the embeddings model."""
    print(f"Embedding model: {embeddings_model.model}")  # text-embedding-3-small
    print(f"Embedding vector size: {len(embeddings_model.embed_query('test'))}")  # 1536
    print(f"Dimensions setting: {embeddings_model.dimensions}")  # None = model default (1536); can be set lower, e.g. 512
    print(f"Max input tokens: {embeddings_model.embedding_ctx_length}")  # 8191: longer texts are split before embedding
    print(f"Batch size: {embeddings_model.chunk_size}\n")  # 1000 texts per API call (not our RAG chunk size!)


def chroma_raw_search(query: str, n_results: int = 3) -> None:
    """Store 3 dummy texts in an in-memory Chroma collection and query it."""
    # No embedding model given: Chroma uses its own local default (all-MiniLM-L6-v2), not OpenAI
    client = chromadb.Client()  # In-memory: data is lost when the script ends
    collection = client.get_or_create_collection(name="test_collection")

    # Dummy texts
    documents = [
        {"id": "doc1", "text": "Hallo, world"},
        {"id": "doc2", "text": "All good today?"},
        {"id": "doc3", "text": "See you around!"},
    ]

    for doc in documents:
        collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

    results = collection.query(
        query_texts=[query],  # Query texts to search for
        n_results=n_results  # Number of results to return
    )

    print(results)


def langchain_chroma_search(query: str, n_results: int = 2) -> None:
    """Store the sample docs in a Chroma DB on disk and run a similarity search."""
    # Create a LangChain vector store. No ids are given, so every run adds all docs again (8 -> 16 -> 24 ...)
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        collection_name="langchain_test_collection",
        persist_directory="data/.chroma_db",  # Save the DB to disk in this folder (kept between runs)
    )
    # Each Document is stored as one row (one chunk). No splitter here, so 1 Document = 1 chunk
    print(f"Rows in DB: {vector_store._collection.count()}")  # Number of stored chunks

    data = vector_store.get(include=["embeddings"])  # By default, get() does not return vectors
    print(f"Vector size: {len(data['embeddings'][0])}")  # Elements in one vector

    # Perform a similarity search
    results = vector_store.similarity_search(query=query, k=n_results)
    print(f"Raw results: {results}")

    print(f"Top {n_results} for query '{query}':")
    for i, doc in enumerate(results):
        print(f"Result {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")


def langchain_chroma_search_with_scores(query: str, n_results: int = 2) -> None:
    """Search the saved Chroma DB and print each result with its distance and similarity score."""
    # Load the existing vector store from disk. No documents are added, so no new duplicates
    vector_store = Chroma(
        collection_name="langchain_test_collection",
        persist_directory="data/.chroma_db",
        embedding_function=embeddings_model
    )
    print(f"Rows in DB: {vector_store._collection.count()}")  # Number of stored chunks

    # Perform a similarity search with scores. Each result is a (Document, score) pair. Score = distance: lower means more similar
    results_with_scores = vector_store.similarity_search_with_score(query=query, k=n_results)
    print(f"Raw results with scores: {results_with_scores}")

    print(f"Top {n_results} for query '{query}' with scores:")
    for i, (doc, score) in enumerate(results_with_scores):
        final_score = 1 / (1 + score)  # Turn distance into similarity: 1 = identical, closer to 0 = less similar
        print(f"Result {i + 1}: {doc.page_content} (Source: {doc.metadata['source']}, Score (distance): {score}, Score (similarity): {final_score})")


def langchain_metadata_filtering(query: str, n_results: int = 5) -> None:
    """Search the saved Chroma DB with and without a metadata filter (topic='database')."""
    # Load the existing vector store from disk. No documents are added, so no new duplicates
    vector_store = Chroma(
        collection_name="langchain_test_collection",
        persist_directory="data/.chroma_db",
        embedding_function=embeddings_model
    )
    print(f"Rows in DB: {vector_store._collection.count()}")  # Number of stored chunks

    # Perform a similarity search without metadata filtering
    filter_criteria = {"topic": "database"}
    results_without_filter = vector_store.similarity_search(query=query, k=n_results)
    print(f"Raw results without metadata filter: {results_without_filter}")
    for i, doc in enumerate(results_without_filter):
        print(f"Result {i + 1}: {doc.page_content} (Source: {doc.metadata['source']}, Topic: {doc.metadata['topic']})")

    # Perform a similarity search with metadata filtering
    results_with_filter = vector_store.similarity_search(query=query, k=n_results, filter=filter_criteria)
    print(f"\nRaw results with metadata filter: {results_with_filter}")

    print(f"\nTop {n_results} for query '{query}' with metadata filter (topic='{filter_criteria['topic']}'):")
    for i, doc in enumerate(results_with_filter):
        print(f"Result {i + 1}: {doc.page_content} (Source: {doc.metadata['source']}, Topic: {doc.metadata['topic']})")


if __name__ == "__main__":
    # chroma_raw_search("Hallo, world")
    # print_embeddings_info()
    # langchain_chroma_search("What is LangChain?")
    # langchain_chroma_search_with_scores("What is LangChain?")
    langchain_metadata_filtering("What is a vector database?")