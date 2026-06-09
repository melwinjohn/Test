from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_pipeline.metadata import DocumentMetadata, infer_document_type, merge_metadata
from rag_pipeline.models import Document

def load_metadata_sidecar(file_path: Path) -> dict[str, Any] | None:
    """Load metadata from invoice.txt.meta.json or invoice.meta.json sidecar files."""
    for candidate in (
        Path(f"{file_path}.meta.json"),
        file_path.with_name(f"{file_path.stem}.meta.json"),
        file_path.with_suffix(".metadata.json"),
    ):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def load_manifest(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Load a manifest mapping sources to metadata.

    Supports JSON:
      {"docs/invoice.txt": {"document_type": "invoice", "contains_pii": true}}
    Or JSON list:
      [{"source": "docs/invoice.txt", "document_type": "invoice"}]
    """
    path = Path(manifest_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}

    if isinstance(raw, list):
        manifest: dict[str, dict[str, Any]] = {}
        for entry in raw:
            source = entry.pop("source", None) or entry.pop("file_path", None)
            if source:
                manifest[str(source)] = entry
        return manifest

    raise ValueError(f"Unsupported manifest format in {path}")


def resolve_manifest_metadata(
    file_path: Path,
    manifest: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not manifest:
        return {}

    candidates = [
        str(file_path),
        str(file_path.resolve()),
        file_path.name,
        str(file_path.as_posix()),
    ]
    for candidate in candidates:
        if candidate in manifest:
            return dict(manifest[candidate])
    return {}


def build_document_metadata(
    file_path: Path,
    *,
    sidecar: dict[str, Any] | None = None,
    manifest_entry: dict[str, Any] | None = None,
    cli_metadata: dict[str, Any] | None = None,
    infer_type: bool = True,
) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    if infer_type:
        doc_type = infer_document_type(str(file_path))
        if doc_type:
            inferred["document_type"] = doc_type.value

    merged = merge_metadata(
        {
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
        },
        inferred,
        sidecar,
        manifest_entry,
        cli_metadata,
    )
    return DocumentMetadata.from_dict(merged).to_dict()


def load_text_file(
    path: str | Path,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
    manifest: dict[str, dict[str, Any]] | None = None,
    infer_type: bool = True,
) -> Document:
    file_path = Path(path)
    sidecar = load_metadata_sidecar(file_path)
    manifest_entry = resolve_manifest_metadata(file_path, manifest)
    doc_metadata = build_document_metadata(
        file_path,
        sidecar=sidecar,
        manifest_entry=manifest_entry,
        cli_metadata=metadata,
        infer_type=infer_type,
    )

    return Document(
        content=file_path.read_text(encoding="utf-8"),
        source=source or str(file_path),
        metadata=doc_metadata,
    )


def load_directory(
    directory: str | Path,
    pattern: str = "*.txt",
    recursive: bool = True,
    metadata: dict[str, Any] | None = None,
    manifest: dict[str, dict[str, Any]] | None = None,
    infer_type: bool = True,
) -> list[Document]:
    root = Path(directory)
    globber = root.rglob if recursive else root.glob
    return [
        load_text_file(
            path,
            metadata=metadata,
            manifest=manifest,
            infer_type=infer_type,
        )
        for path in sorted(globber(pattern))
    ]
