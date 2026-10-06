"""Production-style RAG service on Supabase pgvector: add docs, search with scores, answer with sources.

Run with: uv run src/langchain_rag_playground/rag/supabase_rag_service.py
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()  # Load API keys and the database URL from .env


# Configuration
@dataclass
class Config:
    """Settings for the RAG service: database, models and search."""

    # Database - use pooler URL in production.
    # Note the +psycopg driver prefix (langchain_postgres uses psycopg3).
    database_url: str = os.getenv(
        "SUPABASE_DATABASE_URL",
        os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
        ),
    )
    collection_name: str = "production_documents"

    # Model settings
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Search settings
    default_k: int = 5
    min_similarity: float = 0.5  # Defined in the course, not used yet


class RAGService:
    """RAG service on pgvector: add documents, search with scores, and answer questions."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        # Created on first use (lazy), so creating the service is cheap
        self._vectorstore = None
        self._chain = None

    @property
    def vectorstore(self) -> PGVector:
        """Create the PGVector store on first use."""
        if self._vectorstore is None:
            embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
            self._vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.config.collection_name,
                connection=self.config.database_url,
                use_jsonb=True,  # Store metadata as JSONB (filterable)
            )
        return self._vectorstore

    @property
    def chain(self) -> Runnable:
        """Create the RAG chain on first use."""
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain

    def _create_chain(self) -> Runnable:
        """Create the RAG chain."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.default_k}
        )

        llm = ChatOpenAI(
            model=self.config.chat_model, temperature=self.config.temperature
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer concisely and accurately. If the context doesn't contain relevant information,
say "I don't have enough information to answer that question."
"""
        )

        def format_docs(docs: list[Document]) -> str:
            """Join the retrieved chunks into one context string."""
            return "\n\n".join(doc.page_content for doc in docs)

        # Flow: question -> {context, question} -> prompt -> llm -> plain string
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        """Add documents to the vectorstore."""
        # With fixed ids, re-running upserts instead of creating duplicates
        return self.vectorstore.add_documents(documents, ids=ids)

    def search(self, query: str, k: int | None = None, filter_dict: dict | None = None) -> list[tuple[Document, float]]:
        """Search with optional filtering."""
        search_kwargs = {"k": k or self.config.default_k}
        if filter_dict:
            # Metadata filter, e.g. {"topic": "pgvector"}
            search_kwargs["filter"] = filter_dict

        # Returns (doc, score). Score is cosine DISTANCE: lower = more similar.
        return self.vectorstore.similarity_search_with_score(query, **search_kwargs)

    def ask(self, question: str) -> str:
        """Ask a question using RAG."""
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str) -> dict:
        """Ask a question and return sources."""
        # Get relevant documents
        docs_with_scores = self.search(question)

        # Generate answer (note: the chain retrieves again, a second DB call)
        answer = self.ask(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    # Convert cosine distance to similarity (higher = better)
                    "similarity": 1 - score,
                }
                for doc, score in docs_with_scores
            ],
        }


def main() -> None:
    """Add sample documents, then test search and RAG with sources."""
    print("=" * 60)
    print("Production RAG Service Demo")
    print("=" * 60)

    service = RAGService()

    # Add sample documents
    print("\n📄 Adding sample documents...")
    sample_docs = [
        Document(
            page_content="pgvector is a PostgreSQL extension for vector similarity search. "
            "Enable it with CREATE EXTENSION vector and store embeddings in a vector column.",
            metadata={"source": "docs", "topic": "pgvector"},
        ),
        Document(
            page_content="HNSW indexes provide fast approximate nearest neighbor search. "
            "They trade a little recall for much faster queries on large tables.",
            metadata={"source": "docs", "topic": "indexing"},
        ),
        Document(
            page_content="For production deployments, use connection pooling "
            "(e.g. the Supabase pooler) to avoid running out of database connections.",
            metadata={"source": "docs", "topic": "production"},
        ),
    ]
    ids = service.add_documents(sample_docs, ids=["doc-pgvector", "doc-indexing", "doc-production"])
    print(f"✅ Added {len(ids)} documents")

    # Test search
    print("\n🔍 Testing search...")
    results = service.search("How do I make pgvector faster?", k=2)
    for doc, score in results:
        print(f"   Distance: {score:.4f} - {doc.page_content[:50]}...")

    # Test RAG
    print("\n❓ Testing RAG...")
    question = "What is pgvector and how do I use it?"
    response = service.ask_with_sources(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources used: {len(response['sources'])}")

    print("\n" + "=" * 60)
    print("✅ Production RAG service demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()