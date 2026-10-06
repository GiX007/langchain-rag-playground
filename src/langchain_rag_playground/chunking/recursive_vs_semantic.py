"""Compare recursive and semantic chunking on one document: chunk sizes, topic mixing, and which chunk each test query retrieves.

Run with: uv run src/langchain_rag_playground/chunking/recursive_vs_semantic.py
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # Load API keys from .env


# Sample document with distinct topics
document = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""


# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


# Part A: Check recursive vs semantic chunking

# Step 1: Initial chunking using RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=['\n\n', '\n', '. ', ' ']
)

recursive_chunks = recursive_splitter.split_text(document)

print(f"\nRecursive chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)


# Step 2: Semantic chunking
semantic_chunker = SemanticChunker(
    embeddings_model,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90  # Split at 90th percentile dissimilarity
)

semantic_chunks = semantic_chunker.split_text(document)

print(f"\nSemantic chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)


def detect_topics(chunk: str) -> list[str]:
    """Return the topics a chunk mentions, found by simple keyword matching."""
    topics = []
    lower = chunk.lower()
    if "oauth" in lower or "authenticate" in lower or "token" in lower:
        topics.append("Authentication")
    if "rate limit" in lower or "429" in lower:
        topics.append("Rate Limiting")
    if "error" in lower or "json" in lower:
        topics.append("Error Handling")
    if "webhook" in lower:
        topics.append("Webhooks")
    return topics


def print_chunks(chunks: list[str], method_name: str) -> None:
    """Print chunks with topic detection and clear formatting."""
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks):
        topics = detect_topics(chunk)
        topic_label = f" | Topics: {', '.join(topics)}" if topics else ""
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars){topic_label} ---")
        # Show first 3 lines for a quick preview
        lines = chunk.strip().split("\n")
        preview = "\n".join(lines[:3])
        if len(lines) > 3:
            preview += "\n..."
        print(preview)


# Compare both chunking methods side by side
print_chunks(recursive_chunks, "Recursive Character Chunking")
print_chunks(semantic_chunks, "Semantic Chunking")

# Comparison Summary
#
# Method                        Chunks    Avg Size    Topic Mixing
# ------------------------------------------------------------------------------
# Recursive (fixed-size)        4         302         None
# Semantic (meaning-based)      3         403         1/3 chunks
#
# Key Insight: Recursive splitting uses fixed character limits, so it may split mid-topic or merge unrelated topics. Here, it does a better job.
# Semantic splitting uses embedding similarity to find natural topic boundaries, keeping related content together.


# Part B: Check retrieval quality

# Create two vector stores, one for each chunking method
recursive_vectorstore = Chroma.from_texts(
    recursive_chunks,
    embeddings_model,
    collection_name='recursive_chunks'
)

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks,
    embeddings_model,
    collection_name='semantic_chunks'
)

# Test queries
test_queries = [
    'How do I authenticate with OAuth2?',
    'What happens when I hit the rate limit?',
    'How are webhooks secured?',
    'What format are errors returned in?'
]


def test_retrieval(query: str, vectorstore: Chroma, name: str) -> str:
    """Print and return the top chunk retrieved for a query."""
    results = vectorstore.similarity_search(query, k=1)
    print(f'\n{name} - Query: "{query}"')
    print(f'Retrieved: {results[0].page_content[:150]}...')
    return results[0].page_content


# Run retrieval tests for both methods
print(f"\n{'='*60}")
print("Test Retrieval Quality of Recursive and Semantic Chunks")
print(f"{'='*60}")
for query in test_queries:
    test_retrieval(query, recursive_vectorstore, "Recursive")
    test_retrieval(query, semantic_vectorstore, "Semantic")

# During retrieval quality check we observe that for structured documents (with clear headers/sections), recursive character chunking often retrieves cleaner,
# more focused results than semantic chunking. Semantic chunking tends to perform better on unstructured text where topic shifts are not explicitly marked by headings.