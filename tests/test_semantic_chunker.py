from unittest.mock import MagicMock

from rag_pipeline.semantic_chunker import SemanticChunker


def _mock_settings():
    settings = MagicMock()
    settings.semantic_chunk_breakpoint_percentile = 90.0
    settings.semantic_chunk_max_chars = 2000
    return settings


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Distinct embeddings force breakpoints between every sentence.
        return [[float(index), 0.0, 0.0] for index, _ in enumerate(texts)]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if a[0] == b[0]:
            return 1.0
        return 0.0


def test_find_breakpoints_respects_percentile_threshold(monkeypatch):
    settings = _mock_settings()
    settings.semantic_chunk_breakpoint_percentile = 50.0
    embedder = FakeEmbedder()
    chunker = SemanticChunker(embedder=embedder, settings=settings)

    monkeypatch.setattr(
        embedder,
        "cosine_similarity",
        lambda a, b: 0.9 if a[0] == 0.0 and b[0] == 1.0 else 0.1,
    )

    embeddings = [[0.0], [1.0], [2.0]]
    breakpoints = chunker._find_breakpoints(embeddings)
    assert breakpoints == {2}


def test_group_sentences_at_breakpoints():
    sentences = ["a", "b", "c", "d"]
    groups = SemanticChunker._group_sentences(sentences, {2})
    assert groups == ["a b", "c d"]
