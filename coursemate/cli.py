import argparse
import contextlib
import os
from pathlib import Path
from typing import TextIO

from coursemate.embeddings import Embedder, FakeEmbedder, OllamaEmbedder
from coursemate.ingest import ingest
from coursemate.llm import FakeClient, LLMClient, LLMError, OllamaClient
from coursemate.store import PgVectorStore, StoreError

DEFAULT_DB = "postgresql://coursemate:coursemate@127.0.0.1:5433/coursemate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coursemate")
    parser.add_argument(
        "--provider",
        choices=["ollama", "fake"],
        default=os.environ.get("COURSEMATE_PROVIDER") or "ollama",
    )
    parser.add_argument(
        "--chat-model", default=os.environ.get("COURSEMATE_CHAT_MODEL") or "qwen2.5:3b"
    )
    parser.add_argument(
        "--embed-model", default=os.environ.get("COURSEMATE_EMBED_MODEL") or "nomic-embed-text"
    )
    parser.add_argument(
        "--embed-dim", type=int, default=int(os.environ.get("COURSEMATE_EMBED_DIM") or 768)
    )
    parser.add_argument(
        "--ollama-url", default=os.environ.get("OLLAMA_URL") or "http://localhost:11434"
    )
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL") or DEFAULT_DB)
    parser.add_argument("--table", default=os.environ.get("COURSEMATE_TABLE") or "chunks")

    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subcommands.add_parser("ingest")
    ingest_parser.add_argument("path", type=Path)
    ingest_parser.add_argument("--reset", action="store_true")
    search_parser = subcommands.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("-k", type=int, default=4)
    return parser


def make_models(args: argparse.Namespace) -> tuple[Embedder, LLMClient]:
    if args.provider == "fake":
        return FakeEmbedder(), FakeClient()
    return (
        OllamaEmbedder(model=args.embed_model, dim=args.embed_dim, base_url=args.ollama_url),
        OllamaClient(model=args.chat_model, base_url=args.ollama_url),
    )


def main(argv: list[str], stdout: TextIO, stderr: TextIO) -> int:
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    embedder, _client = make_models(args)
    try:
        store = PgVectorStore(args.db_url, table=args.table)
        try:
            if args.command == "ingest":
                report = ingest(args.path, embedder, store, reset=args.reset)
                print(f"ingested {report.files} files, {report.chunks} chunks", file=stdout)
                for source, count in store.counts().items():
                    print(f"  {source}: {count} chunks", file=stdout)
            else:
                store.prepare(embedder.model, embedder.dim)
                for hit in store.search(embedder.embed_query(args.query), args.k):
                    print(f"{hit.score:.3f}  {hit.source} > {hit.heading}", file=stdout)
                    print(f"       {hit.text[:100]!r}", file=stdout)
        finally:
            store.close()
    except (StoreError, LLMError) as err:
        print(f"error: {err}", file=stderr)
        return 1
    return 0
