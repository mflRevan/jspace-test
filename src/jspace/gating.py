"""Workspace gating of backward passes.

A :class:`WorkspaceGate` leaves the forward pass untouched. During the forward
pass it selects, at each gated layer and token, a set of ``k`` directions; a
tensor hook then replaces the gradient arriving at that block's output by its
projection onto their span:

    g_l  <-  Q_l Q_l^T g_l          (Q_l orthonormal, per token)

so with gates on consecutive layers the backward recursion becomes
``g_l = c_l P_l B_l^T g_{l+1}`` (B_l^T = ordinary backward through block l+1).
With ``confidence=True``, ``c_l`` (per token) is the J-lens readout's probability
mass on its top-k tokens: concentrated workspace access passes gradient,
diffuse access is gated away. Otherwise ``c_l = 1``.

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


_W_EFF_BF16: dict[int, torch.Tensor] = {}


def _w_eff_bf16(lm: LensedModel) -> torch.Tensor:
    """Effective unembedding ``W_U diag(g)`` in bf16 (cached per model)."""
    key = id(lm)
    if key not in _W_EFF_BF16:
        _W_EFF_BF16[key] = (lm.W_U.float() * lm.norm_gain).to(torch.bfloat16)
    return _W_EFF_BF16[key]


@torch.no_grad()
def lens_topk(lm: LensedModel, h: torch.Tensor, layer: int, k: int, *, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` lens tokens ``[T, k]`` and their total lens probability ``[T]``
    for activations ``h`` ``[T, d]``. Uses ``logits = rmsnorm(J h) W_eff^T`` with
    the vocabulary matmul in bf16 (selection and mass only; exact readouts use
    :meth:`LensedModel.lens_logits`)."""
    W = _w_eff_bf16(lm)
    J = lm.J[layer] if layer != lm.n_layers - 1 else None
    eps = getattr(lm.lm._final_norm, "eps", getattr(lm.lm._final_norm, "variance_epsilon", 1e-6))
    ids, mass = [], []
    for s in range(0, h.shape[0], chunk):
        x = h[s : s + chunk].float()
        x = x @ J.T if J is not None else x
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        logits = (x.to(torch.bfloat16) @ W.T).float()
        lse = logits.logsumexp(-1, keepdim=True)
        vals, idx = logits.topk(k, dim=-1)
        ids.append(idx)
        mass.append((vals - lse).exp().sum(-1))
        del logits
    return torch.cat(ids), torch.cat(mass)


@torch.no_grad()
def active_tokens(lm: LensedModel, h: torch.Tensor, layer: int, k: int, rel: float = 0.0) -> torch.Tensor:
    """Top-``k`` lens tokens ``[T, k]`` for activations ``h`` ``[T, d]``. With
    ``rel > 0`` tokens whose lens prob is below ``rel * p_max`` are marked -1."""
    if rel <= 0:
        return lens_topk(lm, h, layer, k)[0]
    logits = lm.lens_logits(h, layer)
    vals, ids = logits.topk(k, dim=-1)
    lp = vals - vals[:, :1]
    return ids.masked_fill(lp < torch.log(torch.tensor(rel)), -1)


@dataclass
class WorkspaceGate:
    lm: LensedModel
    layers: Sequence[int]
    k: int = 10
    mode: str = "jlens"
    rel: float = 0.0
    seed: int = 0
    record_only: bool = False  # select directions but do not modify gradients
    confidence: bool = False  # scale each token's gated gradient by c = lens top-k mass
    conf: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)  # layer -> [T] (if confidence)
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
        if self.confidence:
            # c comes from the J-lens readout in every mode, so controls are matched on c
            act_ids, self.conf[layer] = lens_topk(lm, h.float(), layer, self.k)
        if self.mode == "random":
            V = torch.randn(T, d, self.k, generator=self._g, device=h.device)
            return torch.linalg.qr(V).Q
        if self.mode == "randtok":
            ids = self._pool[torch.randint(len(self._pool), (T, self.k), generator=self._g, device=h.device)]
        elif self.confidence and self.rel <= 0:
            ids = act_ids
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
    @torch.no_grad()
    def _select(self, h: torch.Tensor, layer: int) -> dict:
        """Cheap per-token selection stored for the backward pass (ids, c, seed)."""
        T = h.shape[0]
        sel: dict = {"T": T}
        if self.confidence or self.mode in ("jlens", "rotated"):
            ids, mass = lens_topk(self.lm, h.float(), layer, self.k)
            sel["ids"] = ids
            if self.confidence:
                sel["c"] = mass
                self.conf[layer] = mass
        if self.mode == "randtok":
            sel["ids"] = self._pool[torch.randint(len(self._pool), (T, self.k), generator=self._g, device=h.device)]
        if self.mode == "random":
            sel["seed"] = int(torch.randint(2**31 - 1, (1,), generator=self._g, device=h.device))
        return sel

    def _chunk_vectors(self, sel: dict, layer: int, s: int, e: int, device) -> torch.Tensor:
        """Unit direction vectors ``[n, k, d]`` for tokens ``s:e``."""
        n, d, lm = e - s, self.lm.d_model, self.lm
        if self.mode == "random":
            g = torch.Generator(device=device).manual_seed(sel["seed"] + s)
            V = torch.randn(n, self.k, d, generator=g, device=device)
        else:
            ids = sel["ids"][s:e]
            V = ((lm.W_U[ids.flatten()].float() * lm.norm_gain) @ lm.J[layer]).reshape(n, self.k, d)
            if self.mode == "rotated":
                V = V @ self._rotation(layer).T
        return V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def project(V: torch.Tensor, g: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
        """Orthogonal projection of ``g`` ``[n, d]`` onto span of ``V`` ``[n, k, d]``:
        ``V^T (V V^T + ridge I)^-1 V g`` (a k x k solve per token)."""
        G = V @ V.transpose(1, 2)
        G = G + ridge * torch.eye(G.shape[-1], device=G.device)
        a = torch.linalg.solve(G, torch.einsum("nkd,nd->nk", V, g))
        return torch.einsum("nk,nkd->nd", a, V)

    def _hook(self, layer: int, chunk: int = 4096):
        def fwd(module, inputs, output):
            if torch._C._current_graph_task_id() != -1:
                return  # activation-checkpoint recompute inside backward: already selected
            h = output if torch.is_tensor(output) else output[0]
            B, T, d = h.shape
            sel = self._select(h.detach().reshape(B * T, d), layer)
            if self.record_only or not h.requires_grad:
                return

            def project(g, sel=sel, shape=h.shape):
                flat = g.reshape(-1, shape[-1]).float()
                out = torch.empty_like(flat)
                for s in range(0, flat.shape[0], chunk):
                    e = min(s + chunk, flat.shape[0])
                    out[s:e] = self.project(self._chunk_vectors(sel, layer, s, e, flat.device), flat[s:e])
                if "c" in sel:
                    out = out * sel["c"][:, None]
                return out.reshape(shape).to(g.dtype)

            h.register_hook(project)
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
