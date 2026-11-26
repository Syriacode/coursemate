import json
import statistics

import pytest

from coursemate.stats import format_stats, load_answers, percentile, summarize, write_html

# the ten answer times of guide section 2c, in the order they came in
TIMES = [1200, 900, 15000, 1100, 1000, 950, 1300, 1250, 980, 20000]


@pytest.mark.parametrize(("p", "expected"), [(50, 1100), (90, 15000), (95, 20000)])
def test_the_paper_percentiles(p, expected):
    assert percentile(TIMES, p) == expected


def test_the_paper_mean_is_nobodys_experience():
    mean = statistics.mean(TIMES)

    assert mean == 4368
    assert sum(t < mean for t in TIMES) == 8


@pytest.mark.parametrize(
    ("values", "p", "expected"),
    [([5, 1, 3], 0, 1), ([5, 1, 3], 100, 5), ([7], 50, 7), ([7], 95, 7), ([2, 1], 50, 1)],
)
def test_percentile_edges(values, p, expected):
    assert percentile(values, p) == expected


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_leaves_the_list_alone():
    values = [3, 1, 2]

    percentile(values, 50)

    assert values == [3, 1, 2]


def root(second, ms, *, found=True, cached=False, first=0.0, tokens=(0, 0), cost=0.0, q="Lab 4?"):
    """A root "chat" span, as JsonlSpanExporter writes it."""
    attributes = {
        "langfuse.observation.input": q,
        "coursemate.found": found,
        "coursemate.cached": cached,
        "coursemate.first_token_ms": first,
        "coursemate.input_tokens": tokens[0],
        "coursemate.output_tokens": tokens[1],
    }
    if cost is not None:  # attrs() leaves an unknown cost out
        attributes["coursemate.cost_usd"] = cost
    return {
        "name": "chat",
        "trace_id": f"{second:032x}",
        "span_id": f"{second:016x}",
        "parent_id": None,
        "start": f"2026-11-25T10:00:{second:02d}.000000+00:00",
        "duration_ms": ms,
        "status": "UNSET",
        "attributes": attributes,
    }


MODEL = root(1, 9500.0, first=4000.0, tokens=(1200, 150))
CACHED = root(2, 50.0, cached=True, first=0.2)
REFUSED = root(3, 120.0, found=False, q="Who won the 2022 World Cup?")


def write_log(tmp_path, *records):
    path = tmp_path / "spans.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def log(tmp_path):
    child = {**MODEL, "name": "llm", "span_id": "f" * 16, "parent_id": MODEL["span_id"]}
    # written out of order, with a child span that is not an answer
    return write_log(tmp_path, REFUSED, child, MODEL, CACHED)


def test_load_answers_takes_the_roots_oldest_first(log):
    answers = load_answers(log)

    assert [a.total_ms for a in answers] == [9500.0, 50.0, 120.0]
    assert [a.first_token_ms for a in answers] == [4000.0, None, None]  # only the model call
    assert [a.model_called for a in answers] == [True, False, False]
    assert [a.cached for a in answers] == [False, True, False]
    assert [a.found for a in answers] == [True, True, False]


def test_a_broken_line_is_a_clear_error(tmp_path):
    path = tmp_path / "spans.jsonl"
    path.write_text('{"name": "chat"', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_answers(path)


def test_summarize_the_small_log(log):
    s = summarize(load_answers(log))

    assert (s.n, s.found, s.cached, s.model_calls, s.unfinished) == (3, 2, 1, 1, 0)
    assert (s.rows["all answers"].n, s.rows["all answers"].p50) == (3, 120.0)
    assert (s.rows["not cached"].n, s.rows["not cached"].max) == (2, 9500.0)
    assert (s.rows["cached"].n, s.rows["cached"].p95) == (1, 50.0)
    assert (s.rows["first token"].n, s.rows["first token"].p50) == (1, 4000.0)
    assert (s.mean_input_tokens, s.mean_output_tokens) == (1200, 150)
    assert s.total_cost_usd == 0.0


def test_budgets_ok(log):
    text, ok = format_stats(summarize(load_answers(log)), max_p95_ms=10_000, max_cost_usd=0.001)

    assert ok
    assert "answers: 3   found: 2   cached: 1 (33% hit rate)" in text
    assert "p95 latency 9,500 ms <= 10,000 ms  OK" in text
    assert "cost per answer $0.000000 <= $0.001000  OK" in text
    assert "OVER" not in text


def test_budgets_over(log):
    text, ok = format_stats(summarize(load_answers(log)), max_p95_ms=1000)

    assert not ok
    assert "p95 latency 9,500 ms > 1,000 ms  OVER" in text


def test_hosted_price_counts_only_the_answers_that_called_a_model(log):
    text, ok = format_stats(summarize(load_answers(log)))

    assert ok  # no budgets asked for
    [haiku] = [line for line in text.splitlines() if "claude-haiku-5-5" in line]
    # 1,200 in and 150 out = $0.000195 per model call, and 1 of the 3 answers called the model
    assert "$0.000065 per answer" in haiku
    assert "budgets" not in text


def test_an_unknown_cost_is_never_within_budget(tmp_path):
    unknown = root(4, 800.0, first=300.0, tokens=(10, 5), cost=None)
    s = summarize(load_answers(write_log(tmp_path, MODEL, unknown)))

    text, ok = format_stats(s, max_cost_usd=1.0)

    assert s.total_cost_usd is None
    assert not ok
    assert "cost of these answers: unknown" in text
    assert "OVER" in text


def test_an_unfinished_answer_is_counted_apart(tmp_path):
    failed = {**root(5, 60_000.0), "status": "ERROR", "attributes": {}}
    s = summarize(load_answers(write_log(tmp_path, MODEL, failed)))

    assert (s.n, s.unfinished) == (1, 1)
    assert s.rows["all answers"].max == 9500.0
    assert "unfinished" in format_stats(s)[0]


def test_an_empty_log_fails_its_budgets(tmp_path):
    s = summarize(load_answers(write_log(tmp_path)))

    text, ok = format_stats(s, max_p95_ms=1000, max_cost_usd=1.0)

    assert not ok
    assert "answers: 0" in text and "no answers to measure" in text


def test_the_dashboard_escapes_questions(tmp_path):
    nasty = root(1, 100.0, first=1.0, q='<script>alert("x")</script>')
    answers = load_answers(write_log(tmp_path, nasty, CACHED, REFUSED))
    page = tmp_path / "out" / "dashboard.html"

    write_html(page, answers, summarize(answers))

    text = page.read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    for kind in ("model", "cached", "idk"):
        assert f'class="bar {kind}"' in text
