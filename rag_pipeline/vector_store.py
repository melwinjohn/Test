from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.models import Chunk


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
    ) -> list[dict[str, Any]]:
        source_clause = ""
        params: list[Any] = [query_embedding, query_embedding, top_k]
        if source_filter:
            source_clause = "AND source = %s"
            params = [query_embedding, source_filter, query_embedding, top_k]

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
                    WHERE TRUE {source_clause}
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
