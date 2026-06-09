CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS document_chunks_metadata_gin_idx
ON document_chunks
USING gin (metadata);

CREATE INDEX IF NOT EXISTS document_chunks_document_type_idx
ON document_chunks ((metadata->>'document_type'));

CREATE INDEX IF NOT EXISTS document_chunks_pii_level_idx
ON document_chunks ((metadata->>'pii_level'));

CREATE INDEX IF NOT EXISTS document_chunks_contains_pii_idx
ON document_chunks ((metadata->>'contains_pii'));
