# RAG Ingestion Pipeline

A Python ingestion pipeline for Retrieval-Augmented Generation (RAG) that uses **semantic chunking**, **Cohere embeddings**, and **pgvector** for storage and similarity search.

## Architecture

```mermaid
flowchart LR
    A[Documents] --> B[Sentence Splitter]
    B --> C[Cohere Embeddings]
    C --> D[Semantic Breakpoint Detection]
    D --> E[Chunk Merge / Cap]
    E --> F[Cohere Document Embeddings]
    F --> G[(PostgreSQL + pgvector)]
    H[Query] --> I[Cohere Query Embedding]
    I --> G
    G --> J[Top-K Results]
```

### Semantic chunking

1. Split each document into sentences.
2. Embed sentences with Cohere (`search_document` input type).
3. Compute cosine similarity between adjacent sentences.
4. Start a new chunk when similarity falls below the configured percentile threshold.
5. Merge grouped sentences and enforce a maximum chunk size.

This groups semantically related content instead of splitting on fixed character or token boundaries.

## Prerequisites

- Python 3.11+
- Docker (for local PostgreSQL with pgvector)
- [Cohere API key](https://dashboard.cohere.com/)

## Quick start

### 1. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

### 2. Configure environment

```bash
cp .env.example .env
# Set COHERE_API_KEY in .env
```

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Initialize the database schema

```bash
rag-ingest init-db
```

### 5. Ingest documents

```bash
rag-ingest ingest data/sample.txt
rag-ingest ingest data/ --pattern "*.txt"
```

### 6. Search

```bash
rag-ingest search "How does semantic chunking work?" --top-k 3
```

## Programmatic usage

```python
from rag_pipeline import IngestionPipeline
from rag_pipeline.models import Document

pipeline = IngestionPipeline()
pipeline.initialize()

document = Document(
    content="Your document text here.",
    source="my-doc",
    metadata={"category": "notes"},
)
result = pipeline.ingest_document(document)
print(result)

hits = pipeline.search("your question", top_k=5)
for hit in hits:
    print(hit["similarity"], hit["content"][:120])

pipeline.close()
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `COHERE_API_KEY` | — | Cohere API key (required) |
| `COHERE_EMBED_MODEL` | `embed-english-v3.0` | Cohere embedding model |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag_db` | PostgreSQL connection string |
| `SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE` | `90` | Percentile for semantic breakpoints (higher = fewer, larger chunks) |
| `SEMANTIC_CHUNK_MAX_CHARS` | `2000` | Maximum characters per chunk |
| `EMBED_BATCH_SIZE` | `96` | Batch size for Cohere embed API calls |

## Project layout

```
rag_pipeline/
  config.py           # Settings from environment
  embedder.py         # Cohere embedding client
  semantic_chunker.py # Semantic chunking logic
  vector_store.py     # pgvector persistence and search
  pipeline.py         # End-to-end ingestion orchestration
  loaders.py          # File/directory loaders
  cli.py              # CLI entry point
scripts/init_db.sql   # Optional SQL bootstrap
docker-compose.yml    # Local pgvector database
data/sample.txt       # Example document
```

## Notes

- Cohere `search_document` embeddings are used for stored chunks; `search_query` is used for retrieval queries.
- The `document_chunks` table uses cosine distance (`<=>`) with an IVFFlat index.
- Re-ingesting the same `source` replaces existing chunks for that source.
