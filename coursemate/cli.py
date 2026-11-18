import argparse
import asyncio
import contextlib
import os
from pathlib import Path
from typing import TextIO

import uvicorn

from coursemate.api import create_app
from coursemate.compare import average_tokens, compare, format_costs, format_runs
from coursemate.embeddings import Embedder, FakeEmbedder, OllamaEmbedder
from coursemate.ingest import ingest
from coursemate.llm import FakeClient, LLMClient, LLMError, OllamaClient
from coursemate.rag import DEFAULT_MIN_SCORE, answer
from coursemate.retrieval import MODES, retrieve
from coursemate.retrieval_eval import (
    GoldenSetError,
    build_index,
    evaluate,
    format_table,
    load_golden,
    misses,
    save_run,
)
from coursemate.store import PgVectorStore, StoreError
from coursemate.streaming import make_streamer

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
    search_parser.add_argument("--mode", choices=MODES, default="vector")
    ask_parser = subcommands.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("-k", type=int, default=4)
    ask_parser.add_argument("--mode", choices=MODES, default="vector")
    ask_parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    eval_parser = subcommands.add_parser("eval-retrieval")
    eval_parser.add_argument("notes", type=Path)
    eval_parser.add_argument("golden", type=Path)
    eval_parser.add_argument("--show-misses", action="store_true")
    eval_parser.add_argument("--save", type=Path, metavar="PATH")
    serve_parser = subcommands.add_parser("serve")
    serve_parser.add_argument(
        "--llm",
        choices=["ollama", "openai", "anthropic", "fake"],
        default=None,
        help="which model answers (default: the value of --provider)",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("-k", type=int, default=4)
    serve_parser.add_argument("--mode", choices=MODES, default="vector")
    serve_parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    compare_parser = subcommands.add_parser("compare")
    compare_parser.add_argument("notes", type=Path)
    compare_parser.add_argument(
        "--llm",
        nargs="+",
        choices=["ollama", "openai", "anthropic", "fake"],
        default=["ollama", "openai", "anthropic"],
        help="which providers to compare (default: all three formats)",
    )
    compare_parser.add_argument("-n", type=int, default=3, help="how many golden questions")
    compare_parser.add_argument("--golden", type=Path, default=Path("evals/retrieval_golden.jsonl"))
    return parser


def make_models(args: argparse.Namespace) -> tuple[Embedder, LLMClient]:
    if args.provider == "fake":
        return FakeEmbedder(), FakeClient()
    return (
        OllamaEmbedder(model=args.embed_model, dim=args.embed_dim, base_url=args.ollama_url),
        OllamaClient(model=args.chat_model, base_url=args.ollama_url),
    )


def run_eval(args: argparse.Namespace, embedder: Embedder, stdout: TextIO, stderr: TextIO) -> int:
    """Score vector, keyword and hybrid search on the golden set, in memory (no database)."""
    try:
        golden = load_golden(args.golden)
        store = build_index(args.notes, embedder)
        results = evaluate(golden, embedder, store)
        if args.save:
            save_run(args.save, results, embedder.model)
    except (GoldenSetError, LLMError, OSError) as err:
        print(f"error: {err}", file=stderr)
        return 1

    print(f"embedding model: {embedder.model}", file=stdout)
    print(format_table(results, golden), file=stdout)
    if args.show_misses:
        for line in misses(results, golden):
            print(line, file=stdout)
    if args.save:
        print(f"saved run to {args.save}", file=stdout)
    return 0


def run_compare(
    args: argparse.Namespace, embedder: Embedder, stdout: TextIO, stderr: TextIO
) -> int:
    """Ask the same golden questions through each provider and print times and prices."""
    try:
        # first, so a missing API key stops everything before any indexing or model call
        llms = [make_streamer(name, args.chat_model, args.ollama_url) for name in args.llm]
        golden = load_golden(args.golden)[: args.n]
        if not golden:
            raise ValueError("no golden questions to ask (is -n 0?)")
        store = build_index(args.notes, embedder)  # in memory: no Postgres needed
        runs = asyncio.run(compare(llms, golden, embedder, store))
    except (GoldenSetError, LLMError, OSError, ValueError) as err:
        print(f"error: {err}", file=stderr)
        return 1

    print(format_runs(runs), file=stdout)
    print(file=stdout)
    tokens = average_tokens(runs)
    if tokens is None:
        print("no answer reached a model, so there is nothing to price", file=stdout)
    else:
        print(format_costs(*tokens), file=stdout)
    return 0


def run_serve(args: argparse.Namespace, embedder: Embedder, stderr: TextIO) -> int:
    """Start the web API. It blocks until the server is stopped (Ctrl+C)."""
    try:
        # first, so a bad provider name or a missing API key fails before we touch the database
        llm = make_streamer(args.llm or args.provider, args.chat_model, args.ollama_url)
    except (ValueError, LLMError) as err:
        print(f"error: {err}", file=stderr)
        return 1

    try:
        store = PgVectorStore(args.db_url, table=args.table)
        try:
            # a wrong embedding model fails here, with the usual message
            store.prepare(embedder.model, embedder.dim)
            app = create_app(
                embedder, store, llm, k=args.k, min_score=args.min_score, mode=args.mode
            )
            uvicorn.run(app, host=args.host, port=args.port)
        finally:
            store.close()
    except (StoreError, LLMError) as err:
        print(f"error: {err}", file=stderr)
        return 1
    return 0


def main(argv: list[str], stdout: TextIO, stderr: TextIO) -> int:
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    embedder, client = make_models(args)
    if args.command == "eval-retrieval":
        return run_eval(args, embedder, stdout, stderr)
    if args.command == "compare":
        return run_compare(args, embedder, stdout, stderr)
    if args.command == "serve":
        return run_serve(args, embedder, stderr)

    try:
        store = PgVectorStore(args.db_url, table=args.table)
        try:
            if args.command == "ingest":
                report = ingest(args.path, embedder, store, reset=args.reset)
                print(f"ingested {report.files} files, {report.chunks} chunks", file=stdout)
                for source, count in store.counts().items():
                    print(f"  {source}: {count} chunks", file=stdout)
            elif args.command == "search":
                store.prepare(embedder.model, embedder.dim)
                hits = retrieve(args.query, embedder, store, args.k, args.mode).hits
                for hit in hits:
                    print(f"{hit.score:.3f}  {hit.source} > {hit.heading}", file=stdout)
                    print(f"       {hit.text[:100]!r}", file=stdout)
            else:
                store.prepare(embedder.model, embedder.dim)
                result = answer(
                    args.question,
                    embedder,
                    store,
                    client,
                    k=args.k,
                    min_score=args.min_score,
                    mode=args.mode,
                )
                print(result.answer, file=stdout)
                for number, hit in enumerate(result.sources, start=1):
                    print(
                        f" [{number}] {hit.source} > {hit.heading} ({hit.score:.2f})", file=stdout
                    )
                if result.reason:
                    print(f"(why: {result.reason})", file=stderr)
                print(
                    f"(tokens: {result.input_tokens} in, {result.output_tokens} out)", file=stderr
                )
        finally:
            store.close()
    except (StoreError, LLMError) as err:
        print(f"error: {err}", file=stderr)
        return 1
    return 0
