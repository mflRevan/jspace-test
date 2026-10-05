"""Sparse nonnegative decomposition onto a dictionary (paper sections 2.3, 4.2).

:func:`nn_gradient_pursuit` approximates each row of ``X`` as a nonnegative
combination of at most ``k`` dictionary atoms (gradient pursuit, Blumensath &
Davies 2008, with a nonnegativity projection): each iteration adds the atom
most positively correlated with the residual, then takes one exact line-search
step along the gradient restricted to the current support and clips negative
coefficients to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Decomposition:
    indices: torch.Tensor  # [B, k] atom ids (-1 where no positive atom was found)
    coefs: torch.Tensor  # [B, k] nonnegative coefficients
    recon: torch.Tensor  # [B, d]
    fve: torch.Tensor  # [B, k] fraction of energy explained after each step


@torch.no_grad()
def nn_gradient_pursuit(X: torch.Tensor, D: torch.Tensor, k: int, *, chunk: int = 256) -> Decomposition:
    """``X`` ``[B, d]`` (any float dtype), ``D`` ``[N, d]`` unit-norm atoms (bf16 fine).

    "Energy" is ``||x||^2`` of the rows as given; centre ``X`` beforehand to
    report fraction of variance."""
    out = [_pursuit(X[s : s + chunk].float(), D, k) for s in range(0, X.shape[0], chunk)]
    return Decomposition(*(torch.cat([getattr(o, f) for o in out]) for f in ("indices", "coefs", "recon", "fve")))


def _pursuit(X: torch.Tensor, D: torch.Tensor, k: int) -> Decomposition:
    B, d = X.shape
    idx = torch.full((B, k), -1, dtype=torch.long, device=X.device)
    coef = torch.zeros(B, k, device=X.device)
    S = torch.zeros(B, k, d, device=X.device)  # selected atoms
    recon = torch.zeros_like(X)
    energy = X.pow(2).sum(-1).clamp_min(1e-12)
    fve = torch.zeros(B, k, device=X.device)
    rows = torch.arange(B, device=X.device)
    for it in range(k):
        r = X - recon
        c = (r.to(D.dtype) @ D.T).float()  # [B, N]
        if it:
            sel = idx[:, :it]
            c.scatter_(1, sel.clamp_min(0), float("-inf"))
        best_val, best = c.max(-1)
        ok = best_val > 0
        idx[ok, it] = best[ok]
        S[ok, it] = D[best[ok]].float()
        # gradient step on the support
        g = torch.einsum("bsd,bd->bs", S[:, : it + 1], r)
        Dg = torch.einsum("bs,bsd->bd", g, S[:, : it + 1])
        mu = (r * Dg).sum(-1) / Dg.pow(2).sum(-1).clamp_min(1e-12)
        coef[:, : it + 1] = (coef[:, : it + 1] + mu[:, None] * g).clamp_min(0)
        recon = torch.einsum("bs,bsd->bd", coef[:, : it + 1], S[:, : it + 1])
        fve[:, it] = 1 - (X - recon).pow(2).sum(-1) / energy
    del rows
    return Decomposition(idx, coef, recon, fve)


def random_dictionary(n: int, d: int, *, device, dtype=torch.bfloat16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    out = torch.empty(n, d, dtype=dtype, device=device)
    for s in range(0, n, 32768):
        r = torch.randn(min(32768, n - s), d, device=device, generator=g)
        out[s : s + 32768] = (r / r.norm(dim=-1, keepdim=True)).to(dtype)
    return out
