from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document to ingest."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A semantically derived text chunk ready for embedding."""

    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
