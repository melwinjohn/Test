from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_pipeline.loaders import load_directory, load_manifest, load_text_file
from rag_pipeline.metadata import DocumentMetadata
from rag_pipeline.pipeline import IngestionPipeline
from rag_pipeline.vector_store import MetadataFilter


def _parse_metadata_json(value: str) -> dict:
    return json.loads(value)


def _build_ingest_metadata(args: argparse.Namespace) -> dict:
    layers: list[dict] = []

    if args.metadata:
        layers.append(_parse_metadata_json(args.metadata))
    if args.metadata_file:
        layers.append(json.loads(Path(args.metadata_file).read_text(encoding="utf-8")))

    cli_fields: dict = {}
    if args.doc_type:
        cli_fields["document_type"] = args.doc_type
    if args.document_id:
        cli_fields["document_id"] = args.document_id
    if args.tenant_id:
        cli_fields["tenant_id"] = args.tenant_id
    if args.pii_level:
        cli_fields["pii_level"] = args.pii_level
    if args.access_tier:
        cli_fields["access_tier"] = args.access_tier
    if args.contains_pii:
        cli_fields["contains_pii"] = True
    if args.tags:
        cli_fields["tags"] = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    if args.pii_categories:
        cli_fields["pii_categories"] = [
            item.strip() for item in args.pii_categories.split(",") if item.strip()
        ]

    if cli_fields:
        layers.append(cli_fields)

    if not layers:
        return {}

    merged: dict = {}
    for layer in layers:
        merged.update(layer)
    return DocumentMetadata.from_dict(merged).to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG ingestion pipeline with semantic chunking, Cohere embeddings, and pgvector.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Initialize pgvector schema")
    init_parser.set_defaults(func=cmd_init_db)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory path to ingest")
    ingest_parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern when ingesting a directory (default: *.txt)",
    )
    ingest_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories",
    )
    ingest_parser.add_argument(
        "--manifest",
        help="JSON manifest mapping file paths to document metadata",
    )
    ingest_parser.add_argument(
        "--metadata",
        help='Inline JSON metadata applied to all ingested documents, e.g. \'{"tenant_id":"acme"}\'',
    )
    ingest_parser.add_argument(
        "--metadata-file",
        help="JSON file with metadata applied to all ingested documents",
    )
    ingest_parser.add_argument("--doc-type", choices=["invoice", "contract", "correspondence", "report", "other"])
    ingest_parser.add_argument("--document-id")
    ingest_parser.add_argument("--tenant-id")
    ingest_parser.add_argument("--pii-level", choices=["none", "low", "high", "restricted"])
    ingest_parser.add_argument("--access-tier", help="Access tier label, e.g. legal_only")
    ingest_parser.add_argument("--contains-pii", action="store_true", help="Mark documents as containing PII")
    ingest_parser.add_argument("--tags", help="Comma-separated tags")
    ingest_parser.add_argument("--pii-categories", help="Comma-separated PII categories")
    ingest_parser.add_argument(
        "--no-infer-type",
        action="store_true",
        help="Do not infer document_type from file path",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    search_parser = subparsers.add_parser("search", help="Search ingested chunks")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--source", help="Filter by document source")
    search_parser.add_argument("--doc-type", choices=["invoice", "contract", "correspondence", "report", "other"])
    search_parser.add_argument("--tenant-id")
    search_parser.add_argument("--access-tier")
    search_parser.add_argument("--exclude-pii", action="store_true", help="Exclude chunks marked as containing PII")
    search_parser.add_argument(
        "--pii-level-max",
        choices=["none", "low", "high", "restricted"],
        help="Only return chunks at or below this PII level",
    )
    search_parser.add_argument("--tags", help="Comma-separated tags; chunk must match at least one")
    search_parser.set_defaults(func=cmd_search)

    return parser


def cmd_init_db(_: argparse.Namespace) -> int:
    pipeline = IngestionPipeline()
    try:
        pipeline.initialize()
        print("Database initialized with pgvector schema.")
        return 0
    finally:
        pipeline.close()


def cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.path)
    pipeline = IngestionPipeline()
    ingest_metadata = _build_ingest_metadata(args)
    manifest = load_manifest(args.manifest) if args.manifest else None

    try:
        pipeline.initialize()

        if path.is_file():
            documents = [
                load_text_file(
                    path,
                    metadata=ingest_metadata or None,
                    manifest=manifest,
                    infer_type=not args.no_infer_type,
                )
            ]
        elif path.is_dir():
            documents = load_directory(
                path,
                pattern=args.pattern,
                recursive=not args.no_recursive,
                metadata=ingest_metadata or None,
                manifest=manifest,
                infer_type=not args.no_infer_type,
            )
        else:
            print(f"Path not found: {path}", file=sys.stderr)
            return 1

        if not documents:
            print("No documents found to ingest.", file=sys.stderr)
            return 1

        results = pipeline.ingest_documents(documents)
        for result in results:
            print(
                f"Ingested {result.source}: "
                f"{result.chunks_created} chunks created, "
                f"{result.chunks_upserted} upserted"
            )
        return 0
    finally:
        pipeline.close()


def cmd_search(args: argparse.Namespace) -> int:
    pipeline = IngestionPipeline()
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()] if args.tags else None
    metadata_filter = MetadataFilter(
        document_type=args.doc_type,
        tenant_id=args.tenant_id,
        access_tier=args.access_tier,
        exclude_pii=args.exclude_pii,
        pii_level_max=args.pii_level_max,
        tags=tags,
    )

    try:
        results = pipeline.search(
            query=args.query,
            top_k=args.top_k,
            source_filter=args.source,
            metadata_filter=metadata_filter,
        )
        print(json.dumps(results, indent=2, default=str))
        return 0
    finally:
        pipeline.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
