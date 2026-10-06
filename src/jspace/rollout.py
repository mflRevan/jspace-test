"""Batched rollouts that also record the workspace-gate selection.

During generation, forward hooks on the band layers keep every new activation
(prompt prefill, then one token per decode step). After generation the gate
selection is computed once, in large batches, and stored per token:

  ids   [k]     top-k J-lens tokens (the active J-space)
  c     []      lens probability mass on those tokens (gate confidence)
  inv_n [k]     1 / ||J^T w_i||  (norms of the raw lens vectors)
  ginv  [k, k]  (V V^T + ridge I)^-1 for the unit lens vectors V

so the backward pass of the training forward can apply the gate without any
vocabulary readout (see :class:`jspace.gating.PrecomputedGate`). Selections are
laid out exactly like the returned token tensor ``[B, P + N]`` (left-padded
prompt, then generated tokens); the final token is never fed back, so its
entry is zero (it receives no gradient in training anyway).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from jspace.gating import _w_eff_bf16, lens_topk
from jspace.model import LensedModel


@dataclass
class Selection:
    ids: torch.Tensor  # [B, T, k] long
    c: torch.Tensor  # [B, T]
    inv_n: torch.Tensor  # [B, T, k]
    ginv: torch.Tensor  # [B, T, k, k]

    def rows(self, sl) -> Selection:
        return Selection(self.ids[sl], self.c[sl], self.inv_n[sl], self.ginv[sl])


@dataclass
class Rollout:
    ids: torch.Tensor  # [B, P + N] prompt (left-padded) + generated tokens
    attn: torch.Tensor  # [B, P + N]
    pos: torch.Tensor  # [B, P + N] position ids (attention-mask cumsum)
    n_prompt: int
    gen_mask: torch.Tensor  # [B, N] 1 for generated tokens up to and incl. EOS
    texts: list[str]
    selection: dict[int, Selection] = field(default_factory=dict)


@torch.no_grad()
def select_tokens(lm: LensedModel, h: torch.Tensor, layer: int, k: int, ridge: float = 1e-4):
    """Gate selection for activations ``h`` ``[N, d]``: (ids, c, inv_n, ginv)."""
    ids, c = lens_topk(lm, h, layer, k)
    W = _w_eff_bf16(lm)[ids.flatten()].float()  # [N*k, d]
    V = (W @ lm.J[layer]).view(h.shape[0], k, -1)
    n = V.norm(dim=-1).clamp_min(1e-6)
    U = V / n[..., None]
    G = U @ U.transpose(1, 2) + ridge * torch.eye(k, device=h.device)
    return ids, c, 1.0 / n, torch.linalg.inv(G)


@torch.no_grad()
def rollout(
    lm: LensedModel,
    prompts: list[str],
    *,
    n: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    capture_layers: list[int] | None = None,
    k: int = 10,
    chunk: int = 8192,
) -> Rollout:
    """Sample ``n`` completions per prompt (batched, left padding)."""
    tok = lm.tok
    tok.padding_side = "left"
    texts = [p for p in prompts for _ in range(n)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(lm.device)
    store: dict[int, list] = {l: [] for l in capture_layers or []}

    def hook(layer):
        def fwd(module, inputs, output):
            store[layer].append((output if torch.is_tensor(output) else output[0]).detach())
        return fwd

    # Generation must use the KV cache: in train mode with gradient checkpointing
    # HF silently disables it and re-runs the full sequence every step.
    was_training = lm.hf.training
    lm.hf.eval()
    handles = [lm.layers[l].register_forward_hook(hook(l)) for l in store]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lm.hf.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                 top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id, use_cache=True)
    finally:
        for hd in handles:
            hd.remove()
        lm.hf.train(was_training)
    P = enc.input_ids.shape[1]
    new = out[:, P:]
    eos = torch.tensor([tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id], device=lm.device)
    is_end = torch.isin(new, eos)
    ended_before = (is_end.cumsum(1) - is_end.long()) > 0  # strictly after the first EOS
    gen_mask = (~ended_before).long()
    attn = torch.cat([enc.attention_mask, gen_mask], 1)
    pos = (attn.cumsum(1) - 1).clamp_min(0)
    ro = Rollout(out, attn, pos, P, gen_mask,
                 [tok.decode(r[m.bool()], skip_special_tokens=True) for r, m in zip(new, gen_mask, strict=True)])
    B, T = out.shape
    # Score only real tokens, and each prompt once (its n samples share the prefix).
    live = attn[:, : T - 1].bool().clone()
    if n > 1:
        live[:, :P] &= (torch.arange(B, device=lm.device) % n == 0)[:, None]
    flat_idx = live.flatten().nonzero()[:, 0]
    for l, chunks in store.items():
        h = torch.cat(chunks, 1)  # [B, T - 1, d]: the last generated token was never fed back
        if h.shape[1] != T - 1:
            raise RuntimeError(f"captured {h.shape[1]} positions, expected {T - 1} (KV cache not used?)")
        flat = h.reshape(-1, h.shape[-1])[flat_idx]
        parts = [select_tokens(lm, flat[s : s + chunk], l, k) for s in range(0, flat.shape[0], chunk)]
        ids, c, inv_n, ginv = (torch.cat([p[i] for p in parts]) for i in range(4))
        full = []
        for x in (ids, c, inv_n, ginv):
            buf = torch.zeros(B * T, *x.shape[1:], dtype=x.dtype, device=x.device)
            rows = flat_idx // (T - 1) * T + flat_idx % (T - 1)  # map [B, T-1] index -> [B, T]
            buf[rows] = x
            buf = buf.view(B, T, *x.shape[1:])
            if n > 1:  # copy each prompt's selection to its other samples
                g0 = buf[::n, :P]
                buf[:, :P] = g0.repeat_interleave(n, dim=0)
            full.append(buf)
        ro.selection[l] = Selection(full[0], full[1] * attn, full[2], full[3])
        del h, chunks, flat
    return ro
