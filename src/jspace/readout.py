"""Reading the J-space: lens top-k lists and token ranks over (layer, position).

All ranks are 0-based over the full vocabulary (0 = top), computed exactly by
sorting each readout. ``method="logit"`` gives the logit-lens baseline
(``J = I``) through the same code path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jspace.model import Forward, LensedModel
from jspace.tokens import WordSets, wordlike_mask


class Readout:
    def __init__(self, lm: LensedModel, fw: Forward, *, method: str = "jacobian") -> None:
        self.lm, self.fw, self.method = lm, fw, method
        self.T = fw.ids.shape[0]
        self.layers = list(lm.lens_layers) + [lm.n_layers - 1]

    def _pos(self, positions: Sequence[int] | None) -> list[int]:
        if positions is None:
            return list(range(self.T))
        return [p % self.T for p in positions]

    def logits(self, layer: int, positions: Sequence[int] | None = None) -> torch.Tensor:
        """Lens logits ``[P, vocab]`` (fp32) at ``layer``."""
        h = self.fw.resid[layer, self._pos(positions)]
        return self.lm.lens_logits(h, layer, method=self.method)

    def logprobs(self, layer: int, positions: Sequence[int] | None = None) -> torch.Tensor:
        return self.logits(layer, positions).log_softmax(-1)

    def topk(
        self, layer: int, pos: int, k: int = 10, *, words_only: bool = True
    ) -> list[tuple[str, float]]:
        """Top-``k`` ``(token, logprob)`` at one cell; ``words_only`` filters
        punctuation/fragments from the *display* (probabilities are unchanged)."""
        lp = self.logprobs(layer, [pos])[0]
        if words_only:
            mask = wordlike_mask(self.lm.tok, lp.shape[0]).to(lp.device)
            lp = lp.masked_fill(~mask[: lp.shape[0]], float("-inf"))
        vals, idx = lp.topk(k)
        return [(self.lm.tok.decode([int(i)]), float(v)) for v, i in zip(vals, idx, strict=True)]

    @torch.no_grad()
    def ranks(
        self,
        token_ids: torch.Tensor | Sequence[int],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        chunk: int = 64,
    ) -> torch.Tensor:
        """Ranks ``[L, P, n]`` (int32, CPU) of ``token_ids`` at each (layer, position)."""
        layers = self.layers if layers is None else list(layers)
        pos = self._pos(positions)
        ids = torch.as_tensor(token_ids, device=self.lm.device).long()
        out = torch.empty(len(layers), len(pos), len(ids), dtype=torch.int32)
        for li, layer in enumerate(layers):
            for s in range(0, len(pos), chunk):
                lg = self.logits(layer, pos[s : s + chunk])
                tgt = lg[:, ids]
                srt = lg.sort(dim=-1).values
                ge = torch.searchsorted(srt, tgt.contiguous(), right=True)
                out[li, s : s + chunk] = (lg.shape[-1] - ge).int().cpu()
        return out

    def word_ranks(
        self,
        words: WordSets,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Ranks ``[L, P, n_words]``: per word, the best rank over its surface forms."""
        ids, owner = words.flat()
        r = self.ranks(ids, layers=layers, positions=positions)
        out = torch.full((*r.shape[:2], len(words)), torch.iinfo(torch.int32).max, dtype=torch.int32)
        for j in range(len(words)):
            out[..., j] = r[..., owner == j].min(dim=-1).values
        return out


def readout(lm: LensedModel, text_or_ids, *, method: str = "jacobian") -> Readout:
    return Readout(lm, lm.run(text_or_ids), method=method)


def format_topk(items: list[tuple[str, float]]) -> str:
    return "  ".join(f"{t.strip() or repr(t)}" for t, _ in items)
