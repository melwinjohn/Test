from __future__ import annotations

import cohere
import numpy as np

from rag_pipeline.config import Settings, get_settings


class CohereEmbedder:
    """Embed text using Cohere's embedding models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = cohere.ClientV2(api_key=self.settings.cohere_api_key)
        self.model = self.settings.cohere_embed_model
        self.batch_size = self.settings.embed_batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for storage (search_document input type)."""
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for retrieval (search_query input type)."""
        return self._embed([text], input_type="search_query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
                embedding_types=["float"],
            )
            embeddings.extend(response.embeddings.float_)

        return embeddings

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        vec_a = np.asarray(a, dtype=np.float64)
        vec_b = np.asarray(b, dtype=np.float64)
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)
