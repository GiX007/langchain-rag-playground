"""Connect to Supabase pgvector with LangChain and run a quick add and search test (uses a local Postgres if Supabase is not set).

Run with: uv run src/langchain_rag_playground/vector_stores/supabase_connection.py
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()  # Load API keys and the database URL from .env


# Supabase connection string format (note the +psycopg driver prefix): postgresql+psycopg://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
# Get these from: Supabase Dashboard -> Project Settings -> Database (or the "Connect" button at the top of the dashboard).
# Supabase gives "postgresql://..." -> add "+psycopg" by hand.

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

# For demo, use local if Supabase not configured
DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
)


def connect_to_supabase() -> PGVector:
    """Connect to Supabase pgvector and return a PGVector store."""

    # Embedding model: turns text into vectors (1536 dims for -small)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Note: Supabase has pgvector pre-installed.
    # PGVector creates its own tables on first use and groups documents by collection_name.
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True,  # Store metadata as JSONB (filterable, recommended)
    )

    return vectorstore


def verify_connection(vectorstore: PGVector) -> bool:
    """Add and search one test document. Return True if both work."""

    # Add a test document
    test_doc = Document(
        page_content="This is a test document to verify the pgvector connection.",
        metadata={"test": True},
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"✅ Added test document: {ids[0]}")

        # Search for it
        results = vectorstore.similarity_search("test document", k=1)
        if results:
            print(f"✅ Search works: {results[0].page_content[:60]}...")
        else:
            print("⚠️ Search returned no results")

        # Clean up: remove the test document so the collection stays clean
        # (Commented out, so a new test document is added to the DB on every run)
        # vectorstore.delete(ids)
        # print("✅ Cleanup complete")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main() -> None:
    """Print where we connect, then run the connection test."""
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    if SUPABASE_URL:
        print("\n🔗 Connecting to Supabase...")
        # Print only the host, never the password
        print(f"   Host: {SUPABASE_URL.split('@')[1].split('/')[0]}")
    else:
        print("\n⚠️  SUPABASE_DATABASE_URL not set")
        print("   Using local PostgreSQL instead")
        print("\n   To use Supabase:")
        print("   1. Create project at supabase.com")
        print("   2. Copy the connection string (Project Settings -> Database)")
        print("   3. Add SUPABASE_DATABASE_URL=postgresql+psycopg://... to .env")

    vectorstore = connect_to_supabase()
    ok = verify_connection(vectorstore)

    print("\n" + "=" * 60)
    print("✅ Connection test passed" if ok else "❌ Connection test failed")
    print("=" * 60)


if __name__ == "__main__":
    main()