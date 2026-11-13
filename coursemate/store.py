from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import psycopg
from psycopg import sql

from coursemate.chunking import Chunk
from coursemate.embeddings import Vector, cosine


@dataclass(frozen=True)
class Hit:
    source: str
    heading: str
    text: str
    score: float


class StoreError(Exception):
    """The vector store can't be used as asked (wrong embedding model, unreachable, ...)."""


class VectorStore(Protocol):
    def prepare(self, embed_model: str, dim: int, *, reset: bool = False) -> None: ...

    def replace_source(
        self, source: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]
    ) -> None: ...

    def search(self, vector: Vector, k: int) -> list[Hit]: ...

    def counts(self) -> dict[str, int]: ...

    def all_chunks(self) -> list[Chunk]: ...


def to_pgvector(vec: Vector) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _mismatch(old_model: str, old_dim: int, new_model: str, new_dim: int) -> StoreError:
    return StoreError(
        f"index was built with {old_model} ({old_dim} dims), "
        f"not {new_model} ({new_dim} dims); re-ingest with --reset"
    )


class MemoryStore:
    """Same behaviour as PgVectorStore, in a Python list; for unit tests."""

    def __init__(self) -> None:
        self.rows: list[tuple[Chunk, Vector]] = []
        self.meta: tuple[str, int] | None = None

    def prepare(self, embed_model: str, dim: int, *, reset: bool = False) -> None:
        if reset:
            self.rows = []
            self.meta = None
        if self.meta is not None and self.meta != (embed_model, dim):
            raise _mismatch(*self.meta, embed_model, dim)
        self.meta = (embed_model, dim)

    def replace_source(
        self, source: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]
    ) -> None:
        self.rows = [row for row in self.rows if row[0].source != source]
        self.rows.extend(zip(chunks, vectors, strict=True))

    def search(self, vector: Vector, k: int) -> list[Hit]:
        hits = [Hit(c.source, c.heading, c.text, cosine(vector, v)) for c, v in self.rows]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    def counts(self) -> dict[str, int]:
        counted = Counter(chunk.source for chunk, _ in self.rows)
        return dict(sorted(counted.items()))

    def all_chunks(self) -> list[Chunk]:
        return sorted((chunk for chunk, _ in self.rows), key=lambda c: (c.source, c.index))


class PgVectorStore:
    def __init__(self, url: str, table: str = "chunks") -> None:
        self.table = table
        try:
            self.conn = psycopg.connect(url, autocommit=True, connect_timeout=5)
        except psycopg.OperationalError as exc:
            raise StoreError(
                f"can't connect to Postgres ({exc}). Is docker compose up -d running?"
            ) from exc

    def prepare(self, embed_model: str, dim: int, *, reset: bool = False) -> None:
        table = sql.Identifier(self.table)
        index = sql.Identifier(f"{self.table}_embedding_idx")

        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS index_meta ("
            "table_name text PRIMARY KEY, embed_model text NOT NULL, dim int NOT NULL)"
        )

        if reset:
            self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {table}").format(table=table))
            self.conn.execute("DELETE FROM index_meta WHERE table_name = %s", (self.table,))

        row = self.conn.execute(
            "SELECT embed_model, dim FROM index_meta WHERE table_name = %s", (self.table,)
        ).fetchone()
        if row is not None and (row[0], row[1]) != (embed_model, dim):
            raise _mismatch(row[0], row[1], embed_model, dim)

        self.conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "id bigserial PRIMARY KEY, "
                "source text NOT NULL, "
                "heading text NOT NULL, "
                "content text NOT NULL, "
                "chunk_index int NOT NULL, "
                "embedding vector({dim}) NOT NULL)"
            ).format(table=table, dim=sql.Literal(dim))
        )
        self.conn.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {table} "
                "USING hnsw (embedding vector_cosine_ops)"
            ).format(index=index, table=table)
        )
        self.conn.execute(
            "INSERT INTO index_meta (table_name, embed_model, dim) VALUES (%s, %s, %s) "
            "ON CONFLICT DO NOTHING",
            (self.table, embed_model, dim),
        )

    def replace_source(
        self, source: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]
    ) -> None:
        table = sql.Identifier(self.table)
        delete = sql.SQL("DELETE FROM {table} WHERE source = %s").format(table=table)
        insert = sql.SQL(
            "INSERT INTO {table} (source, heading, content, chunk_index, embedding) "
            "VALUES (%s, %s, %s, %s, %s::vector)"
        ).format(table=table)
        rows = [
            (c.source, c.heading, c.text, c.index, to_pgvector(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute(delete, (source,))
                cur.executemany(insert, rows)

    def search(self, vector: Vector, k: int) -> list[Hit]:
        query = sql.SQL(
            "SELECT source, heading, content, 1 - (embedding <=> %(q)s::vector) AS score "
            "FROM {table} "
            "ORDER BY embedding <=> %(q)s::vector "
            "LIMIT %(k)s"
        ).format(table=sql.Identifier(self.table))
        rows = self.conn.execute(query, {"q": to_pgvector(vector), "k": k}).fetchall()
        return [
            Hit(source, heading, content, float(score)) for source, heading, content, score in rows
        ]

    def counts(self) -> dict[str, int]:
        query = sql.SQL(
            "SELECT source, count(*) FROM {table} GROUP BY source ORDER BY source"
        ).format(table=sql.Identifier(self.table))
        return {source: n for source, n in self.conn.execute(query).fetchall()}

    def all_chunks(self) -> list[Chunk]:
        # COLLATE "C" sorts by raw bytes, like Python, so both stores order sources the same way
        query = sql.SQL(
            "SELECT source, heading, content, chunk_index FROM {table} "
            'ORDER BY source COLLATE "C", chunk_index'
        ).format(table=sql.Identifier(self.table))
        return [
            Chunk(source, heading, index, content)
            for source, heading, content, index in self.conn.execute(query).fetchall()
        ]

    def close(self) -> None:
        self.conn.close()
