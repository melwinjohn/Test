from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_pipeline.loaders import load_directory, load_text_file
from rag_pipeline.pipeline import IngestionPipeline


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
    ingest_parser.set_defaults(func=cmd_ingest)

    search_parser = subparsers.add_parser("search", help="Search ingested chunks")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--source", help="Filter by document source")
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

    try:
        pipeline.initialize()

        if path.is_file():
            documents = [load_text_file(path)]
        elif path.is_dir():
            documents = load_directory(
                path,
                pattern=args.pattern,
                recursive=not args.no_recursive,
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
    try:
        results = pipeline.search(
            query=args.query,
            top_k=args.top_k,
            source_filter=args.source,
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
