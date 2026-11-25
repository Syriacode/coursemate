"""A response cache in front of the model: the same request gets the stored reply."""

import hashlib  # noqa: F401 (for cache_key())
import json  # noqa: F401 (for cache_key())
from collections.abc import Sequence

from coursemate.llm import Message


def cache_key(
    provider: str, model: str, messages: Sequence[Message], temperature: float, max_tokens: int
) -> str:
    """Everything that can change the reply goes into the key, and nothing else."""
    payload = {
        "provider": provider,
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # sort_keys: the same dict written in a different order must give the same key
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
