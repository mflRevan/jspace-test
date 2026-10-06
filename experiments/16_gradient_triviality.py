"""Is J-space gradient capture just the linearised-gradient identity?

To first order g_l = J_local^T W^T delta ~ Jbar^T W^T delta = sum_v delta_v v_v:
the gradient is a delta-weighted sum of J-lens vectors, and delta (dLoss/dlogits)
lives on output tokens. So gating onto J-lens vectors of high-|delta| tokens is
expected to capture gradient trivially. The non-trivial question is whether the
*active* J-space directions that are NOT output tokens (unspoken intermediates)
carry gradient beyond that.

Per layer and token (single gate, k directions each), energy-weighted capture:
  trivial    -- J-lens vectors of the k tokens with largest |S_t|, where
                S_t = sum_{t' >= t} delta_t' (downstream output gradient)
  active     -- top-k active lens tokens (as in experiment 15)
  inter      -- top-k active lens tokens after removing the top-50 |S_t| tokens
                and the answer tokens (intermediates only)
  trivial+inter (union, 2k dims) vs trivial+random (2k dims): extra capture
  randtok    -- k random word-like tokens' J-lens vectors; chance = k/d
Plus linearisation fidelity: cosine between g_l and Jbar_l^T sum_{t'>=t} g_final,t'.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from jspace.data import load_json
from jspace.gating import active_tokens, captured_fraction
from jspace.gradients import seq_logprob_grads
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, shade_band, style
from jspace.tokens import wordlike_mask

GATES = ["trivial", "active", "inter", "randtok", "trivial+inter", "trivial+random"]


def tasks(n_twohop, n_arith, seed):
    items = load_json("experiments/probe-swap.json")["items"][:n_twohop]
    out = [(it["prompt"].rstrip(), " " + it["answer"]) for it in items]
    rng = random.Random(seed)
    for _ in range(n_arith):
        a, b, c, d = rng.randint(1, 9), rng.randint(1, 9), rng.randint(2, 5), rng.randint(1, 9)
        out.append((f"Q: What is ( {a} + {b} ) * {c} - {d}?\nA: The answer is", f" {(a + b) * c - d}"))
    return out


def orth(V: torch.Tensor) -> torch.Tensor:
    """[T, k, d] (rows may be zero) -> orthonormal [T, d, k]."""
    V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.linalg.qr(V.transpose(1, 2)).Q


def jvecs(lm, ids: torch.Tensor, layer: int) -> torch.Tensor:
    T, k = ids.shape
    return ((lm.W_U[ids.flatten()].float() * lm.norm_gain) @ lm.J[layer]).reshape(T, k, -1)


@torch.no_grad()
def analyse(lm, sg, k, pool, gen, acc, layers, answer_ids):
    T = sg.acts.shape[1]
    S = sg.delta.flip(0).cumsum(0).flip(0)  # suffix sums [T, V]
    live = sg.grads[-1].pow(2).sum(-1) > 0
    triv_ids = S.abs().topk(k, dim=-1).indices
    excl = S.abs().topk(50, dim=-1).indices  # [T, 50]
    gfin_suffix = sg.grads[-1].flip(0).cumsum(0).flip(0)  # [T, d]
    for li, l in enumerate(layers):
        g = sg.grads[l]
        energy = g.pow(2).sum(-1) * live
        if energy.sum() == 0:
            continue
        cand = active_tokens(lm, sg.acts[l].float(), l, k + 120)  # [T, k+120]
        bad = (cand[:, :, None] == excl[:, None, :]).any(-1) | torch.isin(cand, answer_ids)
        inter_ids = torch.stack([row[~b][:k] if (~b).sum() >= k else torch.cat([row[~b], row[:k - int((~b).sum())]])
                                 for row, b in zip(cand, bad, strict=True)])
        rand_ids = pool[torch.randint(len(pool), (T, k), generator=gen, device=pool.device)]
        Vt, Va, Vi, Vr = jvecs(lm, triv_ids, l), jvecs(lm, cand[:, :k], l), jvecs(lm, inter_ids, l), jvecs(lm, rand_ids, l)
        Vrand = torch.randn_like(Vt)
        bases = {"trivial": orth(Vt), "active": orth(Va), "inter": orth(Vi), "randtok": orth(Vr),
                 "trivial+inter": orth(torch.cat([Vt, Vi], 1)), "trivial+random": orth(torch.cat([Vt, Vrand], 1))}
        for name, Q in bases.items():
            acc[name][li, 0] += float((captured_fraction(Q, g) * energy).sum())
            acc[name][li, 1] += float(energy.sum())
        # linearisation fidelity
        if l < lm.n_layers - 1:
            ghat = gfin_suffix @ lm.J[l]  # Jbar^T applied row-wise: [T, d]
            num = (g * ghat).sum(-1)
            cos = num / (g.norm(dim=-1) * ghat.norm(dim=-1)).clamp_min(1e-30)
            acc["lin_cos"][li, 0] += float((cos * energy).sum())
            acc["lin_cos"][li, 1] += float(energy.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-twohop", type=int, default=40)
    ap.add_argument("--n-arith", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    layers = lm.lens_layers
    pool = wordlike_mask(lm.tok, lm.vocab_size).nonzero()[:, 0].to(lm.device)
    gen = torch.Generator(device=lm.device).manual_seed(args.seed)
    acc = {s: {g: np.zeros((len(layers), 2)) for g in [*GATES, "lin_cos"]} for s in ("target", "policy")}
    n_policy = 0
    for ti, (prompt, answer) in enumerate(tasks(args.n_twohop, args.n_arith, args.seed)):
        p_ids = lm.encode(prompt)
        n_prompt = p_ids.shape[0]
        a_ids = torch.tensor(lm.tok.encode(answer, add_special_tokens=False), device=lm.device)
        seqs = {"target": [(torch.cat([p_ids, a_ids]), 1.0)]}
        with torch.no_grad():
            out = lm.hf.generate(p_ids[None].expand(args.n_samples, -1), do_sample=True, temperature=1.0,
                                 max_new_tokens=8, pad_token_id=lm.tok.pad_token_id or 0)
        comps = [o[n_prompt:] for o in out]
        r = np.array([lm.tok.decode(c, skip_special_tokens=True).strip().lower().startswith(answer.strip().lower())
                      for c in comps], dtype=float)
        if 0 < r.mean() < 1:
            n_policy += 1
            pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else -1
            seqs["policy"] = [(torch.cat([p_ids, c[c != pad]]), float(a)) for c, a in zip(comps, r - r.mean(), strict=True) if a != 0]
        for src, lst in seqs.items():
            for ids, w in lst:
                sg = seq_logprob_grads(lm, ids, n_prompt, w)
                analyse(lm, sg, args.k, pool, gen, acc[src], layers, a_ids)
        if ti % 10 == 0:
            bi = [layers.index(l) for l in band]
            a = acc["target"]
            print(f"[{ti}] band capture target: " + ", ".join(f"{g}={a[g][bi, 0].sum() / max(a[g][bi, 1].sum(), 1e-30):.4f}" for g in GATES)
                  + f" | lin_cos={a['lin_cos'][bi, 0].sum() / max(a['lin_cos'][bi, 1].sum(), 1e-30):.3f} | policy n={n_policy}", flush=True)

    d = lm.d_model
    bi = [layers.index(l) for l in band]
    early = [layers.index(l) for l in band[:5]]
    summary = {"k": args.k, "chance": args.k / d, "chance_2k": 2 * args.k / d, "n_policy_prompts": n_policy}
    for src in acc:
        ratio = {g: (acc[src][g][:, 0] / np.maximum(acc[src][g][:, 1], 1e-30)).tolist() for g in acc[src]}
        summary[src] = {
            "by_layer": ratio,
            "band": {g: float(acc[src][g][bi, 0].sum() / max(acc[src][g][bi, 1].sum(), 1e-30)) for g in acc[src]},
            "early_band": {g: float(acc[src][g][early, 0].sum() / max(acc[src][g][early, 1].sum(), 1e-30)) for g in acc[src]},
        }
        print(f"\n== {src} (chance k/d={args.k / d:.4f}, 2k/d={2 * args.k / d:.4f})")
        for part in ("early_band", "band"):
            print(f"   {part:<10}", {g: round(v, 4) for g, v in summary[src][part].items()})
        print("   per layer (trivial / inter / lin_cos):",
              " ".join(f"L{l}:{ratio['trivial'][li]:.3f}/{ratio['inter'][li]:.4f}/{ratio['lin_cos'][li]:.2f}" for li, l in enumerate(layers) if l % 3 == 0 or l in band[:1]))

    name = f"16_gradient_triviality/{args.model}"
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    cols = {"trivial": SERIES[1], "active": SERIES[0], "inter": SERIES[2], "randtok": NEUTRAL}
    for g, c in cols.items():
        axes[0].plot(layers, summary["target"]["by_layer"][g], color=c, label=g)
    axes[0].axhline(args.k / d, color=NEUTRAL, ls="--", lw=1, label="chance k/d")
    axes[0].set_yscale("log")
    shade_band(axes[0], (band[0], band[-1]), None)
    axes[0].set_title(f"(a) captured gradient energy, k={args.k} (target)", loc="left")
    axes[0].set_xlabel("layer")
    axes[0].legend(fontsize=7)
    for src, c in (("target", SERIES[0]), ("policy", SERIES[1])):
        byl = summary[src]["by_layer"]
        extra_i = np.array(byl["trivial+inter"]) - np.array(byl["trivial"])
        extra_r = np.array(byl["trivial+random"]) - np.array(byl["trivial"])
        axes[1].plot(layers, extra_i, color=c, label=f"{src}: + intermediates")
        axes[1].plot(layers, extra_r, color=c, ls="--", lw=1, label=f"{src}: + random k")
    shade_band(axes[1], (band[0], band[-1]), None)
    axes[1].set_title("(b) extra capture beyond the trivial gate", loc="left")
    axes[1].set_xlabel("layer")
    axes[1].legend(fontsize=7)
    fig.savefig(out_dir(name) / "gradient_triviality.png")
    save_results(name, {"summary": summary, "layers": layers}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
