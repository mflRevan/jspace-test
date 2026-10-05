"""Workspace gating of backward passes.

A :class:`WorkspaceGate` leaves the forward pass untouched. During the forward
pass it selects, at each gated layer and token, a set of ``k`` directions; a
tensor hook then replaces the gradient arriving at that block's output by its
projection onto their span:

    g_l  <-  Q_l Q_l^T g_l          (Q_l orthonormal, per token)

so with gates on consecutive layers the backward recursion becomes
``g_l = P_l B_l^T g_{l+1}`` (B_l^T = ordinary backward through block l+1).

Direction sets (``mode``):
  "jlens"    -- J-lens vectors of the top-k tokens of the lens readout at that
                activation (the active J-space).
  "random"   -- a fresh random k-dim subspace per token (dimension control).
  "rotated"  -- the selected J-lens vectors under a fixed random rotation per
                layer (same internal geometry, no alignment to the model).
  "randtok"  -- J-lens vectors of k random word-like tokens (J geometry and
                anisotropy, but not *active* tokens).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from jspace.model import LensedModel
from jspace.tokens import wordlike_mask

MODES = ("jlens", "random", "rotated", "randtok")


@torch.no_grad()
def active_tokens(lm: LensedModel, h: torch.Tensor, layer: int, k: int, rel: float = 0.0) -> torch.Tensor:
    """Top-``k`` lens tokens ``[T, k]`` for activations ``h`` ``[T, d]``. With
    ``rel > 0`` tokens whose lens prob is below ``rel * p_max`` are marked -1."""
    logits = lm.lens_logits(h, layer)
    vals, ids = logits.topk(k, dim=-1)
    if rel > 0:
        lp = vals - vals[:, :1]
        ids = ids.masked_fill(lp < torch.log(torch.tensor(rel)), -1)
    return ids


@dataclass
class WorkspaceGate:
    lm: LensedModel
    layers: Sequence[int]
    k: int = 10
    mode: str = "jlens"
    rel: float = 0.0
    seed: int = 0
    record_only: bool = False  # select directions but do not modify gradients
    bases: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)  # layer -> [T, d, k]
    _handles: list = field(default_factory=list, repr=False)
    _rot: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        assert self.mode in MODES, self.mode
        self._g = torch.Generator(device=self.lm.device).manual_seed(self.seed)
        if self.mode == "randtok":
            self._pool = wordlike_mask(self.lm.tok, self.lm.vocab_size).nonzero()[:, 0].to(self.lm.device)

    # ----------------------------------------------------------- directions
    def _rotation(self, layer: int) -> torch.Tensor:
        if layer not in self._rot:
            g = torch.Generator(device=self.lm.device).manual_seed(10_000 + self.seed * 100 + layer)
            A = torch.randn(self.lm.d_model, self.lm.d_model, generator=g, device=self.lm.device)
            self._rot[layer] = torch.linalg.qr(A).Q
        return self._rot[layer]

    @torch.no_grad()
    def basis(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Orthonormal ``[T, d, k]`` basis for activations ``h`` ``[T, d]``."""
        T, d = h.shape
        lm = self.lm
        if self.mode == "random":
            V = torch.randn(T, d, self.k, generator=self._g, device=h.device)
            return torch.linalg.qr(V).Q
        if self.mode == "randtok":
            ids = self._pool[torch.randint(len(self._pool), (T, self.k), generator=self._g, device=h.device)]
        else:
            ids = active_tokens(lm, h.float(), layer, self.k, self.rel)
        safe = ids.clamp_min(0)
        V = (lm.W_U[safe.flatten()].float() * lm.norm_gain) @ lm.J[layer]  # [T*k, d]
        V = V.reshape(T, self.k, d)
        V = V * (ids >= 0)[..., None]  # dropped (below threshold) directions -> 0
        if self.mode == "rotated":
            V = V @ self._rotation(layer).T
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.linalg.qr(V.transpose(1, 2)).Q  # [T, d, k]

    # ------------------------------------------------------------- hooks
    def _hook(self, layer: int):
        def fwd(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            Q = self.basis(h[0].detach(), layer)  # batch size 1
            self.bases[layer] = Q
            if not self.record_only and h.requires_grad:
                h.register_hook(lambda g, Q=Q: torch.einsum("tdk,tk->td", Q, torch.einsum("tdk,td->tk", Q, g[0].float()))[None].to(g.dtype))
        return fwd

    def __enter__(self) -> WorkspaceGate:
        for l in self.layers:
            self._handles.append(self.lm.layers[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc) -> None:
        for hd in self._handles:
            hd.remove()
        self._handles = []


def captured_fraction(Q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Per-token ``||Q Q^T g||^2 / ||g||^2`` for ``Q`` ``[T, d, k]``, ``g`` ``[T, d]``."""
    c = torch.einsum("tdk,td->tk", Q, g.float())
    return c.pow(2).sum(-1) / g.float().pow(2).sum(-1).clamp_min(1e-30)
