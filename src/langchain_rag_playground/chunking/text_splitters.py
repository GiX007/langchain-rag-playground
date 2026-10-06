"""Split text into chunks with RecursiveCharacterTextSplitter, compare chunk sizes, and show why chunk overlap matters.

Run with: uv run src/langchain_rag_playground/chunking/text_splitters.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample texts to split
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

# Not used yet: try it with RecursiveCharacterTextSplitter.from_language(Language.PYTHON)
SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_splitter(text: str = SAMPLE_TEXT, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
    """Split text into chunks and print their count, sizes and the first chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    print(f"Original length: {len(text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]}")
    print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")


def chunk_size_comparison(text: str = SAMPLE_TEXT) -> None:
    """Split the same text with 3 chunk sizes (20% overlap) and print the results."""
    chunk_sizes = [100, 200, 500]

    print("Comparing chunk sizes for RecursiveCharacterTextSplitter")
    for size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 5,  # 20% overlap
        )
        chunks = splitter.split_text(text)
        print(f"\nChunk size: {size}: {len(chunks)} chunks, sizes: {[len(c) for c in chunks]}")


def overlap_importance(text: str = SAMPLE_TEXT) -> None:
    """Split a repeated sentence with and without overlap and print the chunk borders."""
    # Uses its own short text (text2) below, not the text parameter
    # text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeated text
    text2 = "The API expires after 24 hours. You must refresh it using the refresh token. " * 10  # Repeated text

    # Without overlap
    no_overlap = RecursiveCharacterTextSplitter(chunk_size=33, chunk_overlap=0)

    # With overlap
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=33, chunk_overlap=20)

    chunks_no_overlap = no_overlap.split_text(text2)
    chunks_with_overlap = with_overlap.split_text(text2)

    print("Without overlap:")
    print(f"  Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_no_overlap[1][:20]}...")

    # Without overlap, this split is as:
    # Chunk 1: "The API expires after 24 hours."
    # Chunk 2: "You must refresh it using the refresh token."
    # The problem is that if a user asks "How do I refresh the API?", the answer is in chunk 2, and chunk 1 has 
    # no context about the API expiring. The model may match the question with the word 'expires' in chunk 1 and 
    # give an incomplete answer. With overlap, chunk 1 will contain the end of the first sentence and the start 
    # of the second sentence, so the model will have context about both sentences and can give a more complete 
    # answer. So having overlap means the end of one chunk and the start of the next chunk will share some content. 
    # It acts like a cheap insurance policy against losing context (important phrases aren't split at boundaries)
    print("\nWith overlap:")
    print(f"  Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_with_overlap[1][:20]}...")


if __name__ == "__main__":
    # print("Testing RecursiveCharacterTextSplitter")
    # recursive_splitter()
    # chunk_size_comparison()
    overlap_importance()