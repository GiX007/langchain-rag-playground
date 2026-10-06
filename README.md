# LangChain RAG Playground

Small, self-contained scripts to learn RAG (Retrieval-Augmented Generation) with LangChain:
loading, chunking, embeddings, vector stores, retrieval, full RAG pipelines, and production concerns.
Each script tries one idea and prints what happens.


## Contents

- [Setup](#setup)
- [Run](#run)
- [Project structure](#project-structure)
- [Future experiments](#future-experiments)
- [Sources](#sources)


## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Create and activate a virtual environment:

   ```
   uv venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```
   uv pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=...
   ANTHROPIC_API_KEY=...              # only for check_setup.py
   LANGSMITH_API_KEY=...              # only for the production/ scripts
   LANGSMITH_TRACING=false            # scripts that trace turn it on themselves
   SUPABASE_DATABASE_URL=postgresql+psycopg://...   # optional, for the Supabase scripts
   ```

   Never commit `.env` (it is already in `.gitignore`).


## Run

From the project root, run any script with:

```
uv run src/langchain_rag_playground/<folder>/<file>.py
```

Start with `uv run src/langchain_rag_playground/check_setup.py` to check packages and API keys.
Each script's docstring says what it does. Most scripts have several demos; pick one by
uncommenting it at the bottom of the file.

Most scripts call the OpenAI API (gpt-4o-mini, text-embedding-3-small), so each run costs
a little money (usually less than a cent).


## Project structure

Suggested learning order, top to bottom:

| Folder | What it covers |
|---|---|
| `check_setup.py` | Package versions and one test call to OpenAI and Anthropic |
| `loading/` | Load .txt and .pdf files into LangChain Documents |
| `embeddings/` | Chat vs embedding models, similarity, embedding cache |
| `chunking/` | Text splitters, recursive vs semantic chunking, a smart chunker with fallback |
| `vector_stores/` | Chroma (search, scores, metadata filters) and Supabase pgvector |
| `retrieval/` | Hybrid search (BM25 + vectors with RRF), parent-document, compression, multi-query |
| `rag/` | A basic RAG pipeline with sources, and a RAG service on Supabase |
| `production/` | Token budget, caching, JSON logs and metrics, LangSmith tracing |

Other folders:

- `docs/`: sample .txt and .pdf files used by the scripts
- `data/`: local Chroma databases created by the scripts (not committed)


## Future experiments

- Load many files at once: a whole folder (`DirectoryLoader`) with mixed types (PDF, Word, HTML), instead of one .txt and one .pdf.
- Load other sources: YouTube transcripts (`YoutubeLoader`) and web pages (`WebBaseLoader`).
- Handle hard PDFs: scanned pages (OCR), tables and images (e.g. `PyMuPDFLoader`, Unstructured).
- Handle big, badly encoded or broken files without crashing the whole load.
- Build a real knowledge base from the PDFs in `docs/` and ask questions across all of them.
- Keep source and page in metadata and show citations in every answer.
- Tune chunk size and overlap on real documents and measure the effect on retrieval.
- Add a reranker (cross-encoder or Cohere Rerank) after hybrid search.
- Turn the exact-match cache into a real semantic cache (embed the query, match by similarity).
- Try local, open models for chat and embeddings (Ollama, Hugging Face) and compare with OpenAI.
- Add access control so users only retrieve documents they are allowed to see.
- Test retrieval in Greek vs English with multilingual embedding models.
- Build a small eval set (questions + expected answers) and measure retrieval (recall@k) and answer quality (LangSmith evals or RAGAS).
- Update the index when documents change or are deleted, without rebuilding everything (LangChain indexing API).
- Switch LLM providers through config, with retries, timeouts and a fallback model.
- Agentic RAG with LangGraph: let the model decide when to retrieve and rewrite bad queries.
- Pin dependency versions (`pyproject.toml` + `uv.lock`) for reproducible installs.


## Sources

- [LangChain](https://python.langchain.com/)
- [LangSmith](https://docs.langchain.com/langsmith/observability)
- [Chroma](https://docs.trychroma.com/)
- [Supabase pgvector](https://supabase.com/docs/guides/database/extensions/pgvector)
- Production RAG course code by (pdichone): [Part 1](https://github.com/pdichone/production-course-main-code/tree/main), [Part 2](https://github.com/pdichone/fcc-production-rag-part-6)
