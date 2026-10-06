"""On-policy policy-gradient training with optional workspace gating.

One step: sample ``samples`` completions for each of ``prompts_per_step``
prompts (:func:`jspace.rollout.rollout`), reward r in {0, 1}, advantage
A = r - mean(r) per prompt, and minimise the token-level policy-gradient loss

    L = - sum_i A_i sum_t log p(y_it | y_<t, x_i) / (number of scored tokens)

(REINFORCE with a mean baseline; no clipping, KL penalty or reference model).
In the gated arm the gradient reaching every workspace-band layer is projected
onto the active J-space and scaled by the readout confidence, using selections
recorded during the rollout (:class:`jspace.gating.PrecomputedGate`).

Weights stay in bf16; :class:`jspace.optim.KahanSGD` (momentum, Kahan-compensated
bf16 updates) with global gradient clipping. The tied embedding / unembedding and
the final norm are frozen in every arm so the J-lens stays valid.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.checkpoint

from jspace.gating import PrecomputedGate
from jspace.model import LensedModel
from jspace.optim import KahanSGD
from jspace.rollout import rollout


@dataclass
class RLConfig:
    lr: float = 0.1
    momentum: float = 0.9
    clip: float = 1.0
    prompts_per_step: int = 8
    samples: int = 8
    max_new_tokens: int = 512
    temperature: float = 1.0
    micro_batch: int = 8
    gated: bool = False
    k: int = 10


def trainable_params(lm: LensedModel) -> list[torch.nn.Parameter]:
    frozen = {id(lm.lm._embed_tokens.weight), id(lm.lm._lm_head.weight)}
    frozen |= {id(p) for p in lm.lm._final_norm.parameters()}
    params = []
    for p in lm.hf.parameters():
        train = id(p) not in frozen
        p.requires_grad_(train)
        if train:
            params.append(p)
    return params


def _chunk_logprobs(h: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    logits = (h @ w.T).float()
    return -torch.nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)


def token_logprobs(lm: LensedModel, hidden: torch.Tensor, targets: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """log p(targets) from final hidden states ``[B, T, d]`` without materialising
    full-vocabulary logits: the LM head + log-softmax run per position chunk under
    activation checkpointing (logits are recomputed in backward)."""
    w = lm.hf.lm_head.weight
    out = [torch.utils.checkpoint.checkpoint(_chunk_logprobs, hidden[:, s : s + chunk], w, targets[:, s : s + chunk],
                                             use_reentrant=False)
           for s in range(0, hidden.shape[1], chunk)]
    return torch.cat(out, 1)


class RLTrainer:
    def __init__(self, lm: LensedModel, cfg: RLConfig, reward_fn: Callable[[str, str], float]) -> None:
        self.lm, self.cfg, self.reward_fn = lm, cfg, reward_fn
        self.params = trainable_params(lm)
        self.opt = KahanSGD(self.params, lr=cfg.lr, momentum=cfg.momentum)
        lm.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.band = lm.spec.band_layers

    def step(self, prompts: list[str], answers: list[str]) -> dict:
        cfg, lm = self.cfg, self.lm
        t0 = time.time()
        ro = rollout(lm, prompts, n=cfg.samples, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                     capture_layers=self.band if cfg.gated else None, k=cfg.k)
        t_roll = time.time() - t0
        r = np.array([self.reward_fn(t, answers[i // cfg.samples]) for i, t in enumerate(ro.texts)], dtype=np.float32)
        adv = (r.reshape(-1, cfg.samples) - r.reshape(-1, cfg.samples).mean(1, keepdims=True)).reshape(-1)
        active = np.nonzero(adv)[0]
        lens = ro.gen_mask.sum(1).float()
        stats = {"reward": float(r.mean()), "frac_groups_active": float((r.reshape(-1, cfg.samples).std(1) > 0).mean()),
                 "mean_len": float(lens.mean()), "truncated": float((lens >= cfg.max_new_tokens).float().mean()),
                 "t_rollout": t_roll}
        if cfg.gated:
            c = torch.stack([s.c[ro.attn.bool()].float() for s in ro.selection.values()])
            stats["gate_c_mean"] = float(c.mean())
            stats["gate_c_median"] = float(c.median())
        t1 = time.time()
        self.opt.zero_grad(set_to_none=True)
        gnorm = 0.0
        if len(active):
            lm.hf.train()
            n_tok = float(ro.gen_mask[active].sum())
            a_all = torch.tensor(adv, device=lm.device)
            for s in range(0, len(active), cfg.micro_batch):
                rows = torch.tensor(active[s : s + cfg.micro_batch], device=lm.device)
                gate = PrecomputedGate(lm, {l: sel.rows(rows) for l, sel in ro.selection.items()}) if cfg.gated else _Null()
                with gate, torch.autocast("cuda", dtype=torch.bfloat16):
                    h = lm.hf.model(input_ids=ro.ids[rows], attention_mask=ro.attn[rows], position_ids=ro.pos[rows],
                                    use_cache=False).last_hidden_state
                    lp = token_logprobs(lm, h[:, ro.n_prompt - 1 : -1], ro.ids[rows, ro.n_prompt :])
                loss = -(a_all[rows, None] * lp * ro.gen_mask[rows]).sum() / n_tok
                loss.backward()
                del h, lp, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, cfg.clip))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            lm.hf.eval()
        stats.update(grad_norm=gnorm, n_active=int(len(active)), t_train=time.time() - t1)
        return stats


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
