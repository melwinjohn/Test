from __future__ import annotations

import numpy as np

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.embedder import CohereEmbedder
from rag_pipeline.metadata import build_chunk_metadata
from rag_pipeline.models import Chunk, Document
from rag_pipeline.text_splitter import merge_sentences, split_into_sentences


class SemanticChunker:
    """
    Split documents into semantically coherent chunks.

    Sentences are embedded with Cohere, then grouped by detecting breakpoints
    where cosine similarity between adjacent sentences drops below a percentile
    threshold (similar to percentile-based semantic chunking).
    """

    def __init__(
        self,
        embedder: CohereEmbedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or CohereEmbedder(self.settings)
        self.breakpoint_percentile = self.settings.semantic_chunk_breakpoint_percentile
        self.max_chars = self.settings.semantic_chunk_max_chars

    def chunk_document(self, document: Document) -> list[Chunk]:
        sentences = split_into_sentences(document.content)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                self._make_chunk(
                    document=document,
                    content=sentences[0],
                    chunk_index=0,
                    chunk_count=1,
                )
            ]

        embeddings = self.embedder.embed_documents(sentences)
        breakpoints = self._find_breakpoints(embeddings)
        groups = self._group_sentences(sentences, breakpoints)
        merged = merge_sentences(groups, self.max_chars)

        return [
            self._make_chunk(
                document=document,
                content=content,
                chunk_index=index,
                chunk_count=len(merged),
            )
            for index, content in enumerate(merged)
        ]

    def _make_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        chunk_count: int,
    ) -> Chunk:
        metadata = build_chunk_metadata(
            document.metadata,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            content=content,
            file_name=document.metadata.get("file_name"),
            file_path=document.metadata.get("file_path"),
            detect_pii=self.settings.detect_pii_in_chunks,
        )
        return Chunk(
            content=content,
            source=document.source,
            chunk_index=chunk_index,
            metadata=metadata,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def _find_breakpoints(self, embeddings: list[list[float]]) -> set[int]:
        """Return indices after which a new chunk should start."""
        if len(embeddings) < 2:
            return set()

        similarities = [
            self.embedder.cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        threshold = float(np.percentile(similarities, 100 - self.breakpoint_percentile))
        return {index + 1 for index, score in enumerate(similarities) if score < threshold}

    @staticmethod
    def _group_sentences(sentences: list[str], breakpoints: set[int]) -> list[str]:
        groups: list[str] = []
        current: list[str] = []

        for index, sentence in enumerate(sentences):
            current.append(sentence)
            if index + 1 in breakpoints:
                groups.append(" ".join(current))
                current = []

        if current:
            groups.append(" ".join(current))

        return groups
