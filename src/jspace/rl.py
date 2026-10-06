"""Minimal on-policy policy-gradient training with optional workspace gating.

REINFORCE with a per-prompt mean baseline: for each prompt, sample ``n``
completions, reward r_i in {0, 1}, advantage A_i = r_i - mean(r), and minimise
``-sum_i A_i log p(c_i | prompt)`` averaged over the batch. No clipping, KL
penalty or reference model. Optimiser: SGD with momentum and global gradient
clipping (keeps the relative gradient magnitudes the gate produces).

Weights are kept in fp32 with bf16 autocast for compute. The token embedding /
unembedding (tied) and the final norm are frozen in every arm so the J-lens
(defined through W_U and the final norm) stays valid during training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.checkpoint

from jspace.gating import WorkspaceGate
from jspace.model import LensedModel

Task = Callable[[np.random.Generator], tuple[str, str]]  # -> (prompt text, answer)


@dataclass
class RLConfig:
    lr: float = 1e-2
    momentum: float = 0.9
    clip: float = 1.0
    prompts_per_step: int = 8
    samples: int = 8
    max_new_tokens: int = 5
    temperature: float = 1.0
    gate_mode: str | None = None  # None (baseline) or a jspace.gating mode
    gate_k: int = 10


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


def reward(text: str, answer: str) -> float:
    toks = text.strip().split()
    return float(bool(toks) and toks[0].strip(".,") == answer)


class Trainer:
    def __init__(self, lm: LensedModel, cfg: RLConfig, seed: int = 0) -> None:
        self.lm, self.cfg = lm, cfg
        self.params = trainable_params(lm)
        self.opt = torch.optim.SGD(self.params, lr=cfg.lr, momentum=cfg.momentum)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else 0
        self.gate = (WorkspaceGate(lm, lm.spec.band_layers, k=cfg.gate_k, mode=cfg.gate_mode, seed=seed)
                     if cfg.gate_mode else None)

    @torch.no_grad()
    def sample(self, prompt_ids: torch.Tensor) -> list[torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.lm.hf.generate(
                prompt_ids[None].expand(self.cfg.samples, -1), do_sample=True, temperature=self.cfg.temperature,
                top_p=1.0, top_k=0, max_new_tokens=self.cfg.max_new_tokens, pad_token_id=self.pad)
        return [o[prompt_ids.shape[0]:] for o in out]

    def step(self, task: Task) -> dict:
        cfg, lm = self.cfg, self.lm
        self.opt.zero_grad(set_to_none=True)
        rewards, n_active = [], 0
        norm = cfg.prompts_per_step * cfg.samples
        for _ in range(cfg.prompts_per_step):
            prompt, answer = task(self.rng)
            p_ids = lm.encode(prompt)
            comps = self.sample(p_ids)
            r = np.array([reward(lm.tok.decode(c, skip_special_tokens=True), answer) for c in comps])
            rewards.append(r.mean())
            adv = r - r.mean()
            if np.all(adv == 0):
                continue
            n_active += 1
            ids = torch.stack([torch.cat([p_ids, c]) for c in comps])  # equal lengths (generate pads)
            mask = torch.stack([(c != self.pad) for c in comps]).float()
            # keep the first EOS/pad position scored when it is a real EOS token
            n_p = p_ids.shape[0]
            ctx = self.gate if self.gate is not None else _null()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = lm.hf(input_ids=ids, use_cache=False).logits[:, n_p - 1 : -1].float()
            lp = logits.log_softmax(-1).gather(2, ids[:, n_p:, None])[..., 0]
            a = torch.tensor(adv, device=lm.device, dtype=torch.float32)
            loss = -(a[:, None] * lp * mask).sum() / norm
            loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, cfg.clip)) if n_active else 0.0
        if n_active:
            self.opt.step()
        return {"reward": float(np.mean(rewards)), "grad_norm": gnorm, "active_prompts": n_active}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
