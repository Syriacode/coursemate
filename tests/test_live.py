import uuid
from pathlib import Path

import httpx
import pytest
from psycopg import sql

from coursemate.embeddings import OllamaEmbedder
from coursemate.ingest import ingest
from coursemate.llm import OllamaClient
from coursemate.rag import answer
from coursemate.store import PgVectorStore
from tests.conftest import TEST_DB, require_db

pytestmark = pytest.mark.live

SAMPLE = Path(__file__).parents[1] / "notes" / "sample"


@pytest.fixture(scope="module")
def rag():
    try:
        httpx.get("http://localhost:11434/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("Ollama is not running (start it, then: ollama pull qwen2.5:3b)")
    require_db()

    table = f"live_{uuid.uuid4().hex[:8]}"
    embedder = OllamaEmbedder()
    store = PgVectorStore(TEST_DB, table=table)
    try:
        ingest(SAMPLE, embedder, store, reset=True)
        yield embedder, store, OllamaClient(model="qwen2.5:3b")
    finally:
        store.conn.execute(sql.SQL("DROP TABLE IF EXISTS {t}").format(t=sql.Identifier(table)))
        store.conn.execute("DELETE FROM index_meta WHERE table_name = %s", (table,))
        store.close()


def test_answers_from_the_syllabus_with_a_citation(rag):
    result = answer("How many points do I lose for a late project?", *rag)

    assert result.found
    assert "syllabus.md" in [hit.source for hit in result.sources]


def test_says_i_dont_know_for_an_off_topic_question(rag):
    result = answer("Who won the 2022 FIFA World Cup?", *rag)

    assert not result.found
