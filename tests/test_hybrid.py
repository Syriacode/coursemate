import pytest

from coursemate.hybrid import rrf
from coursemate.store import Hit


def hit(name: str, score: float) -> Hit:
    # the same name is the same chunk (source, heading, text), whatever its score
    return Hit(f"{name}.md", name, f"text {name}", score)


def identity(h: Hit) -> tuple[str, str, str]:
    return (h.source, h.heading, h.text)


def test_rrf_fuses_the_worked_example():
    vector = [hit("A", 0.9), hit("B", 0.8)]
    keyword = [hit("B", 7.5), hit("C", 3.2)]

    fused = rrf([vector, keyword])

    assert [identity(h) for h in fused] == [
        identity(hit("B", 0)),
        identity(hit("A", 0)),
        identity(hit("C", 0)),
    ]
    assert [h.score for h in fused] == pytest.approx([1 / 62 + 1 / 61, 1 / 61, 1 / 62])


def test_rrf_ignores_the_original_scores():
    first = [hit("X", 0.51), hit("Y", 0.50)]
    second = [hit("Y", 900.0), hit("X", 800.0)]

    fused = rrf([first, second])

    assert len(fused) == 2
    assert [h.score for h in fused] == pytest.approx([1 / 61 + 1 / 62, 1 / 61 + 1 / 62])


def test_rrf_of_empty_lists_is_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []
