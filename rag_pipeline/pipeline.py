from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.embedder import CohereEmbedder
from rag_pipeline.models import Chunk, Document
from rag_pipeline.semantic_chunker import SemanticChunker
from rag_pipeline.vector_store import PgVectorStore


@dataclass
class IngestionResult:
    source: str
    chunks_created: int
    chunks_upserted: int


class IngestionPipeline:
    """End-to-end RAG ingestion: semantic chunk -> Cohere embed -> pgvector store."""

    def __init__(
        self,
        settings: Settings | None = None,
        chunker: SemanticChunker | None = None,
        embedder: CohereEmbedder | None = None,
        vector_store: PgVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or CohereEmbedder(self.settings)
        self.chunker = chunker or SemanticChunker(self.embedder, self.settings)
        self.vector_store = vector_store or PgVectorStore(self.settings)

    def initialize(self) -> None:
        self.vector_store.initialize()

    def ingest_document(self, document: Document, replace: bool = True) -> IngestionResult:
        if replace:
            self.vector_store.delete_by_source(document.source)

        chunks = self.chunker.chunk_document(document)
        upserted = self._embed_and_store(chunks)

        return IngestionResult(
            source=document.source,
            chunks_created=len(chunks),
            chunks_upserted=upserted,
        )

    def ingest_documents(
        self,
        documents: list[Document],
        replace: bool = True,
    ) -> list[IngestionResult]:
        return [self.ingest_document(document, replace=replace) for document in documents]

    def ingest_text(
        self,
        content: str,
        source: str,
        metadata: dict | None = None,
        replace: bool = True,
    ) -> IngestionResult:
        document = Document(content=content, source=source, metadata=metadata or {})
        return self.ingest_document(document, replace=replace)

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None):
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_filter=source_filter,
        )

    def close(self) -> None:
        self.vector_store.close()

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        return self.vector_store.upsert_chunks(chunks, embeddings)
