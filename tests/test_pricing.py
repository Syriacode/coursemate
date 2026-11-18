import dataclasses

import pytest

from coursemate.pricing import FREE, PRICES, PRICES_CHECKED, Price, cost_usd, price_for


def test_cost_matches_the_paper_example():
    # 700 tokens in, 60 out
    assert cost_usd(700, 60, Price(2.00, 10.00)) == pytest.approx(0.002)
    assert cost_usd(700, 60, Price(0.10, 0.50)) == pytest.approx(0.0001)


def test_free_costs_nothing():
    assert cost_usd(700, 60, FREE) == 0


def test_price_for_is_free_when_local():
    assert price_for("qwen2.5:3b", local=True) == FREE
    # the Anthropic format pointed at local Ollama must never be priced as hosted
    assert price_for("claude-sonnet-5-5", local=True) == FREE


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-haiku-5-5", Price(0.10, 0.50)),
        ("claude-sonnet-5-5", Price(2.00, 10.00)),
        ("gpt-6-luna", Price(0.10, 0.50)),
        ("gpt-6.1-sol", Price(2.00, 10.00)),
    ],
)
def test_price_for_a_known_hosted_model(model, expected):
    assert price_for(model, local=False) == expected


def test_price_for_an_unknown_hosted_model_is_none():
    assert price_for("qwen2.5:3b", local=False) is None


def test_the_table_has_the_four_models_and_a_check_date():
    assert set(PRICES) == {"claude-haiku-5-5", "claude-sonnet-5-5", "gpt-6-luna", "gpt-6.1-sol"}
    assert PRICES_CHECKED == "2026-11-15"


def test_price_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FREE.input_per_mtok = 1.0  # type: ignore[misc]
