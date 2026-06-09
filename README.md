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

### 6. Ingest documents with metadata (invoices, contracts, PII)

Use sidecar files, a manifest, or CLI flags:

```bash
# Sidecar: data/samples/invoice.txt + invoice.txt.meta.json
rag-ingest ingest data/samples/

# Manifest maps file paths to metadata
rag-ingest ingest data/samples/ --manifest data/manifest.json

# CLI flags for batch metadata
rag-ingest ingest data/invoices/ \
  --doc-type invoice \
  --tenant-id acme-corp \
  --contains-pii \
  --pii-level high \
  --tags billing,accounts-payable
```

### 7. Search with metadata filters

```bash
rag-ingest search "payment terms" --doc-type contract --exclude-pii
rag-ingest search "invoice total" --doc-type invoice --pii-level-max low
rag-ingest search "termination clause" --access-tier legal_only
```

## Metadata model

Each chunk inherits document metadata and adds chunk-level fields:

| Field | Level | Purpose |
|-------|-------|---------|
| `document_type` | document | `invoice`, `contract`, `correspondence`, `report`, `other` |
| `document_id` | document | Business identifier (invoice #, contract ID) |
| `tenant_id` | document | Tenant / customer scope |
| `contains_pii` | document | Whether the chunk is marked as containing PII |
| `pii_level` | document | `none`, `low`, `high`, `restricted` |
| `pii_categories` | document | e.g. `email`, `ssn`, `bank_account` |
| `access_tier` | document | Access control hint (`legal_only`, `finance_only`) |
| `tags` | document | Free-form labels for filtering |
| `chunk_index` | chunk | Position within the source document |
| `chunk_count` | chunk | Total chunks for the source |
| `detected_pii_categories` | chunk | PII patterns found in chunk text (optional scan) |

Metadata is stored as JSONB on every row in `document_chunks` and indexed for filtering.

### Sidecar metadata file

`invoice.txt.meta.json` next to `invoice.txt`:

```json
{
  "document_type": "invoice",
  "document_id": "INV-2024-0847",
  "tenant_id": "acme-corp",
  "contains_pii": true,
  "pii_level": "high",
  "pii_categories": ["email", "bank_account"],
  "access_tier": "finance_only",
  "tags": ["accounts-payable"]
}
```

Set `DETECT_PII_IN_CHUNKS=true` to scan chunk text for common PII patterns (email, phone, SSN, etc.) and enrich `detected_pii_categories`.

## Programmatic usage

```python
from rag_pipeline import IngestionPipeline
from rag_pipeline.models import Document

pipeline = IngestionPipeline()
pipeline.initialize()

document = Document(
    content="Your document text here.",
    source="my-doc",
    metadata={
        "document_type": "contract",
        "tenant_id": "acme-corp",
        "contains_pii": True,
        "pii_level": "low",
        "access_tier": "legal_only",
    },
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
| `DETECT_PII_IN_CHUNKS` | `true` | Scan chunk text for PII patterns |

## Project layout

```
rag_pipeline/
  config.py           # Settings from environment
  embedder.py         # Cohere embedding client
  semantic_chunker.py # Semantic chunking logic
  vector_store.py     # pgvector persistence and search
  pipeline.py         # End-to-end ingestion orchestration
  loaders.py          # File/directory loaders + manifest/sidecar metadata
  metadata.py         # Metadata schema, PII detection, chunk enrichment
  cli.py              # CLI entry point
scripts/init_db.sql   # Optional SQL bootstrap
docker-compose.yml    # Local pgvector database
data/sample.txt       # Example document
```

## Notes

- Cohere `search_document` embeddings are used for stored chunks; `search_query` is used for retrieval queries.
- The `document_chunks` table uses cosine distance (`<=>`) with an IVFFlat index.
- Re-ingesting the same `source` replaces existing chunks for that source.
