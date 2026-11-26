"""Latency, cost, hit rate and budgets, computed from the JSONL span log."""

import html
import json
import math  # noqa: F401 (for percentile())
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from coursemate.pricing import PRICES, PRICES_CHECKED, cost_usd


def percentile(values: Sequence[float], p: float) -> float:
    """The nearest-rank percentile p (0 to 100): sort the values, then take the one at
    rank ceil(p / 100 * n), counting from 1.

    p = 0 gives the smallest value. An empty list raises ValueError. The input isn't changed.
    """
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))  # 1-based: the rank-th smallest value
    return ordered[max(rank, 1) - 1]


@dataclass(frozen=True)
class Answer:
    """One root "chat" span."""

    start: str
    question: str
    total_ms: float
    found: bool
    cached: bool
    failed: bool  # the model failed or the reader left, so there is no outcome
    first_token_ms: float | None  # only for answers that called the model
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None: unknown

    @property
    def model_called(self) -> bool:
        return self.first_token_ms is not None


@dataclass(frozen=True)
class Row:
    n: int
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class Summary:
    n: int  # finished answers
    found: int
    cached: int
    model_calls: int
    unfinished: int  # failed or abandoned, left out of every other number
    rows: dict[str, Row | None]  # None: there is nothing to measure
    mean_input_tokens: float | None  # per model call
    mean_output_tokens: float | None
    total_cost_usd: float | None  # None when any answer's cost is unknown


def _answer(record: dict) -> Answer:
    attributes = record["attributes"]
    failed = "coursemate.found" not in attributes
    cached = attributes.get("coursemate.cached", False)
    first_token = attributes.get("coursemate.first_token_ms", 0.0)
    # a refusal before the model has no first token; a cached reply has one, but no model call
    called = not failed and not cached and first_token > 0
    return Answer(
        start=record["start"],
        question=attributes.get("langfuse.observation.input", ""),
        total_ms=record["duration_ms"],
        found=attributes.get("coursemate.found", False),
        cached=cached,
        failed=failed,
        first_token_ms=first_token if called else None,
        input_tokens=attributes.get("coursemate.input_tokens", 0),
        output_tokens=attributes.get("coursemate.output_tokens", 0),
        # no model call costs nothing, whatever the model's price is
        cost_usd=attributes.get("coursemate.cost_usd") if called else 0.0,
    )


def load_answers(path: str | Path) -> list[Answer]:
    """The answers in a span log (root "chat" spans), oldest first."""
    answers = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record["name"] != "chat" or record["parent_id"] is not None:
                    continue  # only roots are answers
                answers.append(_answer(record))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path} line {line_number}: not a span record: {exc!r}") from exc
    return sorted(answers, key=lambda a: a.start)


def _row(values: list[float]) -> Row | None:
    if not values:
        return None
    return Row(len(values), percentile(values, 50), percentile(values, 95), max(values))


def summarize(answers: Sequence[Answer]) -> Summary:
    done = [a for a in answers if not a.failed]
    calls = [a for a in done if a.model_called]
    costs = [a.cost_usd for a in done]
    return Summary(
        n=len(done),
        found=sum(a.found for a in done),
        cached=sum(a.cached for a in done),
        model_calls=len(calls),
        unfinished=len(answers) - len(done),
        rows={
            "all answers": _row([a.total_ms for a in done]),
            "not cached": _row([a.total_ms for a in done if not a.cached]),
            "cached": _row([a.total_ms for a in done if a.cached]),
            "first token": _row([a.first_token_ms for a in calls]),
        },
        mean_input_tokens=statistics.mean(a.input_tokens for a in calls) if calls else None,
        mean_output_tokens=statistics.mean(a.output_tokens for a in calls) if calls else None,
        total_cost_usd=None if None in costs else sum(costs),
    )


def _budgets(s: Summary, max_p95_ms: float | None, max_cost_usd: float | None) -> list[tuple]:
    """(text, passed) for each budget asked for. Anything we can't measure fails."""
    results = []
    if max_p95_ms is not None:
        row = s.rows["all answers"]
        if row is None:
            results.append(("p95 latency: no answers to measure", False))
        else:
            passed = row.p95 <= max_p95_ms
            sign = "<=" if passed else ">"
            results.append((f"p95 latency {row.p95:,.0f} ms {sign} {max_p95_ms:,.0f} ms", passed))
    if max_cost_usd is not None:
        if s.n == 0:
            results.append(("cost per answer: no answers to measure", False))
        elif s.total_cost_usd is None:
            results.append(("cost per answer unknown (a model has no price)", False))
        else:
            each = s.total_cost_usd / s.n
            passed = each <= max_cost_usd
            sign = "<=" if passed else ">"
            results.append((f"cost per answer ${each:.6f} {sign} ${max_cost_usd:.6f}", passed))
    return results


def format_stats(
    summary: Summary, max_p95_ms: float | None = None, max_cost_usd: float | None = None
) -> tuple[str, bool]:
    """The report, and False when any budget is over."""
    s = summary
    rate = f" ({s.cached / s.n:.0%} hit rate)" if s.n else ""
    lines = [f"answers: {s.n}   found: {s.found}   cached: {s.cached}{rate}"]
    if s.unfinished:
        lines.append(f"unfinished (failed or abandoned, not counted below): {s.unfinished}")

    lines += ["", f"{'':<14}{'n':>4}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}"]
    for name, row in s.rows.items():
        if row is None:
            lines.append(f"{name:<14}{0:>4}{'-':>10}{'-':>10}{'-':>10}")
        else:
            lines.append(f"{name:<14}{row.n:>4}{row.p50:>10,.0f}{row.p95:>10,.0f}{row.max:>10,.0f}")

    lines.append("")
    if s.mean_input_tokens is None:
        lines.append("no answer called a model, so there are no tokens to price")
    else:
        tokens_in, tokens_out = s.mean_input_tokens, s.mean_output_tokens
        lines.append(f"tokens per model call: {tokens_in:,.0f} in, {tokens_out:,.0f} out")
        lines.append(f"if hosted (prices checked {PRICES_CHECKED}; cached answers cost nothing):")
        for model, price in PRICES.items():
            each = cost_usd(tokens_in, tokens_out, price)  # one model call
            per_answer = each * s.model_calls / s.n  # cached and refused answers are free
            lines.append(
                f"  {model:<20}${per_answer:.6f} per answer   ${per_answer * 1000:.2f} per 1,000"
            )
    total = "unknown" if s.total_cost_usd is None else f"${s.total_cost_usd:.6f}"
    lines.append(f"cost of these answers: {total}")

    results = _budgets(s, max_p95_ms, max_cost_usd)
    if results:
        lines += ["", "budgets:"]
        lines += [f"  {text}  {'OK' if passed else 'OVER'}" for text, passed in results]
    return "\n".join(lines), all(passed for _, passed in results)


_PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CourseMate stats</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fff; color: #222; }}
  pre {{ overflow-x: auto; }}
  .row {{ display: flex; gap: 0.5rem; align-items: center; margin: 2px 0; }}
  .label {{ width: 22rem; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
  .track {{ flex: 1; }}
  .bar {{ height: 1rem; min-width: 2px; }}
  .model {{ background: #2563eb; }}
  .cached {{ background: #16a34a; }}
  .idk {{ background: #dc2626; }}
  .ms {{ width: 6rem; text-align: right; }}
</style>
<h1>CourseMate stats</h1>
<pre>{report}</pre>
<h2>One bar per answer, oldest first</h2>
<p>green: cached · blue: not cached · red: "I don't know"</p>
{bars}
</html>
"""


def write_html(path: str | Path, answers: Sequence[Answer], summary: Summary) -> None:
    """A one-page dashboard: the report, and one bar per finished answer."""
    done = [a for a in answers if not a.failed]
    longest = max((a.total_ms for a in done), default=0) or 1
    bars = []
    for a in done:
        kind = "idk" if not a.found else "cached" if a.cached else "model"
        # questions are typed by users: escape them, or they could inject HTML into this page
        question = html.escape(a.question)
        bars.append(
            f'<div class="row"><span class="label" title="{question}">{question}</span>'
            f'<span class="track"><div class="bar {kind}" '
            f'style="width: {100 * a.total_ms / longest:.1f}%"></div></span>'
            f'<span class="ms">{a.total_ms:,.0f} ms</span></div>'
        )
    report, _ = format_stats(summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _PAGE.format(report=html.escape(report), bars="\n".join(bars)), encoding="utf-8"
    )
