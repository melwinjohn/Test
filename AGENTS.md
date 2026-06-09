# Agent context

## Ingestion operations

- The RAG ingestion pipeline runs as a **nightly batch job** (CRON), not as a real-time or on-demand API.
- Design for: idempotent reruns, failure recovery, off-peak Cohere/DB usage, and batch PII redaction before embed.
- Daytime systems consume the vector index built overnight; ingestion latency of hours is acceptable.

## Corpus

- Mixed document types: invoices, contracts, correspondence, etc.
- Corpus contains **PII**; metadata tagging alone is insufficient — redact before chunking/embedding in batch.

## Stack

- Semantic chunking → Cohere embeddings → pgvector (PostgreSQL)
- Retrieval / full RAG generation not yet implemented
