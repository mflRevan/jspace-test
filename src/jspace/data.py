"""Datasets: Anthropic's released prompt sets (vendored in ``data/``) and
held-out pretraining-like text."""

from __future__ import annotations

import json
from functools import lru_cache

from jspace.config import DATA_DIR


def load_json(rel: str) -> dict:
    return json.loads((DATA_DIR / rel).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def heldout_passages(n: int = 64, *, min_chars: int = 600, split: str = "test") -> tuple[str, ...]:
    """First ``n`` WikiText-103 records of >= ``min_chars`` from ``split``.

    The lenses are fit on the *train* split, so the default ``test`` split is
    held out from lens fitting."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    out: list[str] = []
    for rec in ds:
        t = rec["text"]
        if len(t.strip()) >= min_chars:
            out.append(t.strip())
            if len(out) == n:
                break
    return tuple(out)
