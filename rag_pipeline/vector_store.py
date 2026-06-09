from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.metadata import PiiLevel
from rag_pipeline.models import Chunk


class MetadataFilter:
    """Filter chunks by metadata fields during similarity search."""

    def __init__(
        self,
        document_type: str | None = None,
        tenant_id: str | None = None,
        access_tier: str | None = None,
        exclude_pii: bool = False,
        pii_level_max: PiiLevel | str | None = None,
        tags: list[str] | None = None,
        contains: dict[str, Any] | None = None,
    ) -> None:
        self.document_type = document_type
        self.tenant_id = tenant_id
        self.access_tier = access_tier
        self.exclude_pii = exclude_pii
        self.pii_level_max = (
            PiiLevel(pii_level_max) if isinstance(pii_level_max, str) else pii_level_max
        )
        self.tags = tags or []
        self.contains = contains or {}

    def to_sql(self) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if self.document_type:
            clauses.append("metadata->>'document_type' = %s")
            params.append(self.document_type)

        if self.tenant_id:
            clauses.append("metadata->>'tenant_id' = %s")
            params.append(self.tenant_id)

        if self.access_tier:
            clauses.append("metadata->>'access_tier' = %s")
            params.append(self.access_tier)

        if self.exclude_pii:
            clauses.append("(metadata->>'contains_pii')::boolean IS NOT TRUE")

        if self.pii_level_max is not None:
            allowed = [level.value for level in PiiLevel if _pii_rank(level) <= _pii_rank(self.pii_level_max)]
            clauses.append("metadata->>'pii_level' = ANY(%s)")
            params.append(allowed)

        if self.tags:
            clauses.append("metadata->'tags' ?| %s")
            params.append(self.tags)

        for key, value in self.contains.items():
            clauses.append("metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))

        return clauses, params


def _pii_rank(level: PiiLevel) -> int:
    from rag_pipeline.metadata import PII_LEVEL_RANK

    return PII_LEVEL_RANK[level]


class PgVectorStore:
    """Persist and query chunk embeddings in PostgreSQL with pgvector."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dimensions = self.settings.embedding_dimensions
        self.pool = ConnectionPool(
            conninfo=self.settings.database_url,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
        )

    def initialize(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id BIGSERIAL PRIMARY KEY,
                        source TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (source, chunk_index)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_chunks_metadata_gin_idx
                    ON document_chunks
                    USING gin (metadata)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_chunks_document_type_idx
                    ON document_chunks ((metadata->>'document_type'))
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_chunks_pii_level_idx
                    ON document_chunks ((metadata->>'pii_level'))
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS document_chunks_contains_pii_idx
                    ON document_chunks ((metadata->>'contains_pii'))
                    """
                )
            conn.commit()

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0

        rows = [
            (
                chunk.source,
                chunk.chunk_index,
                chunk.content,
                json.dumps(chunk.metadata),
                embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_chunks (
                        source, chunk_index, content, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (source, chunk_index)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,
                    rows,
                )
            conn.commit()

        return len(rows)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[dict[str, Any]]:
        where_clauses = ["TRUE"]
        params: list[Any] = [query_embedding]

        if source_filter:
            where_clauses.append("source = %s")
            params.append(source_filter)

        if metadata_filter:
            metadata_clauses, metadata_params = metadata_filter.to_sql()
            where_clauses.extend(metadata_clauses)
            params.extend(metadata_params)

        params.extend([query_embedding, top_k])
        where_sql = " AND ".join(where_clauses)

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id,
                        source,
                        chunk_index,
                        content,
                        metadata,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM document_chunks
                    WHERE {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    params,
                )
                return list(cur.fetchall())

    def delete_by_source(self, source: str) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE source = %s",
                    (source,),
                )
                deleted = cur.rowcount
            conn.commit()
        return deleted

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        with self.pool.connection() as conn:
            register_vector(conn)
            yield conn
