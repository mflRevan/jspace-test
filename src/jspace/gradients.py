"""Residual-stream gradients of sequence log-likelihoods (no parameter grads)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jlens.hooks import ActivationRecorder

from jspace.model import LensedModel


@dataclass
class SeqGrad:
    grads: torch.Tensor  # [L, T, d] dLoss/dh_l (fp32)
    acts: torch.Tensor  # [L, T, d] detached activations (model dtype)
    delta: torch.Tensor  # [T, V] dLoss/dlogits (fp32; zero outside the scored span)


def seq_logprob_grads(lm: LensedModel, ids: torch.Tensor, n_prompt: int, weight: float = 1.0) -> SeqGrad:
    """Gradients of ``loss = -weight * log p(ids[n_prompt:] | ids[:n_prompt])`` with
    respect to every block output, with all parameters frozen."""
    L = lm.n_layers
    with torch.enable_grad(), ActivationRecorder(lm.layers, at=range(L), start_graph_at=0) as rec:
        logits = lm.hf(input_ids=ids[None], use_cache=False).logits[0].float()
        acts = [rec.activations[l] for l in range(L)]
        lp = logits[n_prompt - 1 : -1].log_softmax(-1)
        tgt = ids[n_prompt:]
        loss = -weight * lp.gather(1, tgt[:, None]).sum()
        grads = torch.autograd.grad(loss, acts)
    delta = torch.zeros_like(logits)
    with torch.no_grad():
        sm = lp.exp()
        sm[torch.arange(len(tgt)), tgt] -= 1.0
        delta[n_prompt - 1 : -1] = weight * sm
    return SeqGrad(torch.stack([g[0].float() for g in grads]), torch.stack([a.detach()[0] for a in acts]), delta)
