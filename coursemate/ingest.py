from dataclasses import dataclass
from pathlib import Path

from coursemate.chunking import chunk_document
from coursemate.embeddings import Embedder, Vector
from coursemate.store import VectorStore

SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class IngestReport:
    files: int
    chunks: int


def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)


def ingest(
    path: Path | str,
    embedder: Embedder,
    store: VectorStore,
    *,
    reset: bool = False,
    batch_size: int = 16,
) -> IngestReport:
    path = Path(path)
    store.prepare(embedder.model, embedder.dim, reset=reset)

    files = find_files(path)
    total_chunks = 0
    for file in files:
        source = file.name if path.is_file() else file.relative_to(path).as_posix()
        chunks = chunk_document(source, file.read_text(encoding="utf-8"))
        texts = [f"{c.heading}\n{c.text}" if c.heading else c.text for c in chunks]

        vectors: list[Vector] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(embedder.embed_documents(texts[start : start + batch_size]))

        store.replace_source(source, chunks, vectors)
        total_chunks += len(chunks)

    return IngestReport(files=len(files), chunks=total_chunks)
