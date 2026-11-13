import math

import pytest

from coursemate.chunking import Chunk
from coursemate.keyword import KeywordIndex, bm25_term_score, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("When is Lab 4 due?") == ["lab", "4", "due"]


def test_bm25_matches_the_worked_example():
    score = bm25_term_score(tf=2, df=1, n_docs=3, doc_len=4, avg_len=4)

    assert score == pytest.approx(math.log(1 + 2.5 / 1.5) * 2 * 2.5 / (2 + 1.5))
    assert score == pytest.approx(1.40, abs=0.01)


def test_rare_words_beat_common_words():
    rare = bm25_term_score(tf=1, df=1, n_docs=10, doc_len=4, avg_len=4)
    common = bm25_term_score(tf=1, df=5, n_docs=10, doc_len=4, avg_len=4)

    assert rare > common


def test_repeating_a_word_helps_but_less_than_proportionally():
    once = bm25_term_score(tf=1, df=1, n_docs=3, doc_len=4, avg_len=4)
    twice = bm25_term_score(tf=2, df=1, n_docs=3, doc_len=4, avg_len=4)

    assert once < twice < 2 * once


def test_a_longer_chunk_scores_lower_for_the_same_count():
    short = bm25_term_score(tf=1, df=1, n_docs=3, doc_len=4, avg_len=4)
    long = bm25_term_score(tf=1, df=1, n_docs=3, doc_len=8, avg_len=4)

    assert long < short


def test_keyword_index_ranks_the_worked_example():
    index = KeywordIndex(
        [
            Chunk("c1.md", "", 0, "Lab 4 is due Friday"),
            Chunk("c2.md", "", 0, "Labs are worth 30%"),
            Chunk("c3.md", "", 0, "Project 2 is due week 13"),
        ]
    )

    hits = index.search("When is Lab 4 due?", k=3)

    assert [h.source for h in hits] == ["c1.md", "c3.md"]
    assert hits[0].score == pytest.approx(2.43, abs=0.01)
    assert hits[1].score == pytest.approx(0.42, abs=0.01)
