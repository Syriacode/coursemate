from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """What a model charges, in dollars per million tokens (MTok)."""

    input_per_mtok: float
    output_per_mtok: float


FREE = Price(0.0, 0.0)  # a local model: no per-token charge

# Prices copied from the providers' pricing pages on this date. They change often:
# RE-CHECK THEM before you quote a cost to anyone.
PRICES_CHECKED = "2026-11-15"
PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "gpt-6-luna": Price(0.10, 0.50),
    "gpt-6.1-sol": Price(2.00, 10.00),
}


def cost_usd(input_tokens: int, output_tokens: int, price: Price) -> float:
    """Dollars for one answer: GUIDE-api.md section 3g."""
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


def price_for(model: str, local: bool) -> Price | None:
    """The price of a model, or None when we don't know it."""
    if local:
        return FREE  # nothing leaves this computer, even if the model name looks like a hosted one
    return PRICES.get(model)  # None: not in the table, so the page shows "cost unknown"
