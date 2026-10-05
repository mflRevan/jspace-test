"""Forward passes and greedy decoding under J-space interventions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from jspace.interventions import Edit, Intervene
from jspace.model import LensedModel


@torch.no_grad()
def next_logits(lm: LensedModel, ids: torch.Tensor | str, edits: Iterable[Edit] = ()) -> torch.Tensor:
    """Final-position next-token logits ``[vocab]`` (fp32) with ``edits`` active."""
    if isinstance(ids, str):
        ids = lm.encode(ids)
    with Intervene(lm, edits):
        out = lm.hf(input_ids=ids[None], use_cache=False, logits_to_keep=1)
    return out.logits[0, -1].float()


def top_tokens(lm: LensedModel, logits: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
    lp = logits.log_softmax(-1)
    v, i = lp.topk(k)
    return [(lm.tok.decode([int(t)]), round(float(x), 3)) for x, t in zip(v, i, strict=True)]


def rank_of(logits: torch.Tensor, token_ids: Iterable[int]) -> int:
    """Best 0-based rank among ``token_ids`` in ``logits``."""
    ids = torch.as_tensor(list(token_ids), device=logits.device)
    best = logits[ids].max()
    return int((logits > best).sum())


@dataclass
class Generation:
    prompt_ids: torch.Tensor
    new_ids: torch.Tensor
    text: str


@torch.no_grad()
def generate(
    lm: LensedModel,
    ids: torch.Tensor | str,
    edits: Iterable[Edit] = (),
    *,
    max_new_tokens: int = 40,
    stop_ids: Iterable[int] | None = None,
) -> Generation:
    """Greedy decoding with a KV cache; edits see consistent absolute positions."""
    if isinstance(ids, str):
        ids = lm.encode(ids)
    stop = set(stop_ids) if stop_ids is not None else _default_stops(lm)
    new: list[int] = []
    with Intervene(lm, edits) as iv:
        iv.offset = 0
        out = lm.hf(input_ids=ids[None], use_cache=True, logits_to_keep=1)
        cache, cur = out.past_key_values, ids.shape[0]
        for _ in range(max_new_tokens):
            nxt = int(out.logits[0, -1].argmax())
            new.append(nxt)
            if nxt in stop:
                break
            iv.offset = cur
            out = lm.hf(
                input_ids=torch.tensor([[nxt]], device=lm.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache, cur = out.past_key_values, cur + 1
    new_t = torch.tensor(new, device=lm.device)
    return Generation(ids, new_t, lm.tok.decode(new, skip_special_tokens=True))


def _default_stops(lm: LensedModel) -> set[int]:
    stops = set()
    for name in ("eos_token_id", "pad_token_id"):
        t = getattr(lm.tok, name, None)
        if t is not None:
            stops.add(int(t))
    for s in ("<|im_end|>", "<|endoftext|>"):
        t = lm.tok.convert_tokens_to_ids(s)
        if isinstance(t, int) and t >= 0:
            stops.add(t)
    return stops


@torch.no_grad()
def continuation_logprobs(
    lm: LensedModel, prompt: torch.Tensor | str, continuations: list[str], edits: Iterable[Edit] = ()
) -> list[float]:
    """Total log-prob of each continuation string given ``prompt`` with ``edits``
    active at every position (prompt and continuation alike, i.e. clamped).

    Scoring whole strings makes answers comparable across tokenizations (Qwen
    splits " 8" into " " + "8" and "Basketball" into "Basket" + "ball")."""
    if isinstance(prompt, str):
        prompt = lm.encode(prompt)
    n = prompt.shape[0]
    out = []
    with Intervene(lm, edits):
        for cont in continuations:
            c = torch.tensor(lm.tok.encode(cont, add_special_tokens=False), device=lm.device)
            ids = torch.cat([prompt, c])[None]
            lp = lm.hf(input_ids=ids, use_cache=False).logits[0, n - 1 : -1].float().log_softmax(-1)
            out.append(float(lp.gather(1, c[:, None]).sum()))
    return out
