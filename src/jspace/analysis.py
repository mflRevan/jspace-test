"""Helpers for summarising readouts over the workspace band."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jspace.model import LensedModel
from jspace.readout import Readout
from jspace.tokens import wordlike_mask


def find_token(lm: LensedModel, text: str, needle: str, *, occurrence: int = -1, which: str = "last") -> int:
    """Index of the token covering the first/last character of ``needle``'s
    ``occurrence``-th appearance in ``text`` (tokenized as by ``lm.encode``)."""
    starts, i = [], text.find(needle)
    while i >= 0:
        starts.append(i)
        i = text.find(needle, i + 1)
    if not starts:
        raise ValueError(f"{needle!r} not in text")
    s = starts[occurrence]
    char = s + len(needle) - 1 if which == "last" else s
    enc = lm.tok(text, return_offsets_mapping=True, add_special_tokens=True)
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= char < b:
            return t
    raise ValueError("offset not found")


@torch.no_grad()
def band_logprobs(ro: Readout, positions: Sequence[int], layers: Sequence[int]) -> torch.Tensor:
    """Lens log-probs ``[L, P, vocab]`` over ``layers`` at ``positions``."""
    return torch.stack([ro.logprobs(l, positions) for l in layers])


@torch.no_grad()
def band_top(
    ro: Readout,
    pos: int,
    layers: Sequence[int],
    k: int = 12,
    *,
    agg: str = "median",
    words_only: bool = True,
    exclude: Sequence[int] = (),
) -> list[tuple[str, float]]:
    """Top-``k`` tokens at one position by the median (or max) over ``layers``
    of the lens log-prob (paper Fig. 44 uses the median over the band)."""
    lp = band_logprobs(ro, [pos], layers)[:, 0]
    score = lp.median(0).values if agg == "median" else lp.max(0).values
    if words_only:
        mask = wordlike_mask(ro.lm.tok, score.shape[0]).to(score.device)
        score = score.masked_fill(~mask, float("-inf"))
    if exclude:
        score[list(exclude)] = float("-inf")
    v, idx = score.topk(k)
    return [(ro.lm.tok.decode([int(i)]).strip(), round(float(x), 2)) for x, i in zip(v, idx, strict=True)]


@torch.no_grad()
def band_hits(
    ro: Readout, token_ids: Sequence[int], layers: Sequence[int], *, k: int = 10, positions: Sequence[int] | None = None
) -> torch.Tensor:
    """Bool ``[P, n]``: token reaches lens top-``k`` at any band layer."""
    r = ro.ranks(token_ids, layers=layers, positions=positions)
    return (r < k).any(0)
