from __future__ import annotations

from pathlib import Path

from rag_pipeline.models import Document


def load_text_file(path: str | Path, source: str | None = None) -> Document:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    return Document(
        content=content,
        source=source or str(file_path),
        metadata={"file_name": file_path.name, "file_path": str(file_path.resolve())},
    )


def load_directory(
    directory: str | Path,
    pattern: str = "*.txt",
    recursive: bool = True,
) -> list[Document]:
    root = Path(directory)
    globber = root.rglob if recursive else root.glob
    return [load_text_file(path) for path in sorted(globber(pattern))]
