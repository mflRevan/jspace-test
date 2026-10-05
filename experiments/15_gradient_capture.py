"""How much of a learning gradient lies in the active J-space?

Feasibility measurement for workspace-gated RL. Gradient sources (no training):
  target -- d/dθ of -log p(correct answer | prompt)
  policy -- on-policy REINFORCE with a mean baseline over n samples:
            -sum_i (r_i - mean r) log p(sample_i | prompt), r = answer correct
Tasks: two-hop factual prompts (probe-swap.json) and no-CoT arithmetic.

A. Single gate. For every layer l, the gradient g_l at the block output is
   projected per token onto k directions; we report the energy-weighted captured
   fraction sum ||P g||^2 / sum ||g||^2 for the gate modes of jspace.gating
   (jlens, random, rotated, randtok), k in {5, 10, 25, 50}. Chance = k/d.
B. Successive gates on every band layer (k = 10). Per layer below/within the
   band: norm ratio and cosine of the gated vs ungated residual gradient, and
   per-block weight-gradient norm ratio (exact) and cosine (estimated on a fixed
   1/64 coordinate subsample).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from jlens.hooks import ActivationRecorder

from jspace.data import load_json
from jspace.gating import MODES, WorkspaceGate, captured_fraction
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, shade_band, style

KS = [5, 10, 25, 50]
COLORS = {"jlens": SERIES[0], "rotated": SERIES[1], "randtok": SERIES[2], "random": NEUTRAL}


def tasks(n_twohop: int, n_arith: int, seed: int) -> list[tuple[str, str]]:
    items = load_json("experiments/probe-swap.json")["items"][:n_twohop]
    out = [(it["prompt"].rstrip(), " " + it["answer"]) for it in items]
    rng = random.Random(seed)
    for _ in range(n_arith):
        a, b, c, d = rng.randint(1, 9), rng.randint(1, 9), rng.randint(2, 5), rng.randint(1, 9)
        out.append((f"Q: What is ( {a} + {b} ) * {c} - {d}?\nA: The answer is", f" {(a + b) * c - d}"))
    return out


def fwd_bwd(lm, ids: torch.Tensor, n_prompt: int, weight: float, gate: WorkspaceGate | None):
    """Forward prompt+continuation, backward ``-weight * log p(continuation)``.
    Returns residual gradients ``[L, T, d]`` (post-gate) and detached activations;
    parameter gradients accumulate in ``.grad``."""
    L = lm.n_layers
    with ActivationRecorder(lm.layers, at=range(L)) as rec:
        if gate is not None:
            with gate:
                logits = lm.hf(input_ids=ids[None], use_cache=False).logits[0]
        else:
            logits = lm.hf(input_ids=ids[None], use_cache=False).logits[0]
        acts = [rec.activations[l] for l in range(L)]
        for a in acts:
            a.retain_grad()
        lp = logits[n_prompt - 1 : -1].float().log_softmax(-1)
        tgt = ids[n_prompt:]
        loss = -weight * lp.gather(1, tgt[:, None]).sum()
        loss.backward()
    grads = torch.stack([a.grad[0].float() for a in acts])
    hs = torch.stack([a.detach()[0] for a in acts])
    return grads, hs


def block_param_grads(lm, stride: int = 64) -> list[tuple[float, torch.Tensor]]:
    """Per block: exact gradient norm and a fixed strided subsample of the
    flattened gradient (for cosine estimates without storing full copies)."""
    out = []
    for blk in lm.layers:
        gs = [p.grad.flatten() for p in blk.parameters() if p.grad is not None]
        norm = float(torch.sqrt(sum(g.float().pow(2).sum() for g in gs)))
        sub = torch.cat([g[::stride].float() for g in gs])
        out.append((norm, sub))
    return out


def zero_grads(lm):
    for p in lm.hf.parameters():
        p.grad = None


@torch.no_grad()
def sample(lm, ids, n, max_new=8, temp=1.0):
    out = lm.hf.generate(ids[None].expand(n, -1), do_sample=True, temperature=temp, top_p=1.0,
                         max_new_tokens=max_new, pad_token_id=lm.tok.pad_token_id or 0)
    return [o[ids.shape[0]:] for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-twohop", type=int, default=40)
    ap.add_argument("--n-arith", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--k-successive", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, args.lens)
    for p in lm.hf.parameters():
        p.requires_grad_(True)
    band = lm.spec.band_layers
    L, d = lm.n_layers, lm.d_model
    lens_layers = lm.lens_layers

    # A: energy accumulators [source][mode][k] -> per-layer (captured, total)
    cap = {s: {m: {k: np.zeros((len(lens_layers), 2)) for k in KS} for m in MODES} for s in ("target", "policy")}
    # B: successive gating
    succ = {s: {m: {"resid_ratio": [], "resid_cos": [], "param_ratio": [], "param_cos": []} for m in MODES}
            for s in ("target", "policy")}
    n_policy = 0
    probe_gates = {m: {k: WorkspaceGate(lm, lens_layers, k=k, mode=m, seed=args.seed) for k in KS} for m in MODES}

    for ti, (prompt, answer) in enumerate(tasks(args.n_twohop, args.n_arith, args.seed)):
        p_ids = lm.encode(prompt)
        n_prompt = p_ids.shape[0]
        seqs = {"target": [(torch.cat([p_ids, torch.tensor(lm.tok.encode(answer, add_special_tokens=False), device=lm.device)]), 1.0)]}
        comps = sample(lm, p_ids, args.n_samples)
        texts = [lm.tok.decode(c, skip_special_tokens=True) for c in comps]
        r = np.array([t.strip().lower().startswith(answer.strip().lower()) for t in texts], dtype=float)
        if 0 < r.mean() < 1:
            adv = r - r.mean()
            seqs["policy"] = [(torch.cat([p_ids, c[c != (lm.tok.pad_token_id or -1)]]), float(a)) for c, a in zip(comps, adv, strict=True) if a != 0]
            n_policy += 1
        for src, seq_list in seqs.items():
            # ungated reference
            zero_grads(lm)
            ref_g, hs_all = [], []
            for ids, w in seq_list:
                g, hs = fwd_bwd(lm, ids, n_prompt, w, None)
                ref_g.append(g)
                hs_all.append(hs)
            ref_p = block_param_grads(lm)
            # A: single-gate capture at every lens layer
            for g, hs in zip(ref_g, hs_all, strict=True):
                for li, l in enumerate(lens_layers):
                    gl, energy = g[l], g[l].pow(2).sum(-1)
                    for m in MODES:
                        for k in KS:
                            Q = probe_gates[m][k].basis(hs[l], l)
                            frac = captured_fraction(Q, gl)
                            cap[src][m][k][li, 0] += float((frac * energy).sum())
                            cap[src][m][k][li, 1] += float(energy.sum())
            # B: successive gates across the band
            for m in MODES:
                zero_grads(lm)
                gate = WorkspaceGate(lm, band, k=args.k_successive, mode=m, seed=args.seed)
                gg = [fwd_bwd(lm, ids, n_prompt, w, gate)[0] for ids, w in seq_list]
                gp = block_param_grads(lm)
                rr, rc = [], []
                for l in range(L):
                    a = torch.cat([x[l].flatten() for x in gg])
                    b = torch.cat([x[l].flatten() for x in ref_g])
                    rr.append(float(a.norm() / b.norm().clamp_min(1e-30)))
                    rc.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
                pr = [a[0] / max(b[0], 1e-30) for a, b in zip(gp, ref_p, strict=True)]
                pc = [float(torch.nn.functional.cosine_similarity(a[1], b[1], dim=0)) for a, b in zip(gp, ref_p, strict=True)]
                for key, val in (("resid_ratio", rr), ("resid_cos", rc), ("param_ratio", pr), ("param_cos", pc)):
                    succ[src][m][key].append(val)
        if ti % 10 == 0:
            j = cap["target"]
            bi = [lens_layers.index(l) for l in band]
            msg = {m: round(float(j[m][10][bi, 0].sum() / j[m][10][bi, 1].sum()), 4) for m in MODES}
            print(f"[{ti}] band capture (target, k=10): {msg}  policy prompts so far: {n_policy}", flush=True)

    zero_grads(lm)
    bi = [lens_layers.index(l) for l in band]
    summary = {"chance_k_over_d": {k: k / d for k in KS}, "n_policy_prompts": n_policy}
    for src in cap:
        summary[src] = {
            "band_capture": {m: {k: float(cap[src][m][k][bi, 0].sum() / max(cap[src][m][k][bi, 1].sum(), 1e-30)) for k in KS} for m in MODES},
            "capture_by_layer_k10": {m: (cap[src][m][10][:, 0] / np.maximum(cap[src][m][10][:, 1], 1e-30)).tolist() for m in MODES},
            "successive": {m: {key: np.mean(np.array(v), axis=0).tolist() for key, v in succ[src][m].items() if v} for m in MODES},
        }
    for src in ("target", "policy"):
        print(f"\n== {src}: band-mean captured fraction (chance = k/d)")
        for m in MODES:
            print(f"   {m:<8}", {k: round(summary[src]["band_capture"][m][k], 4) for k in KS})
        s = summary[src]["successive"]
        lo = band[0] - 1
        for m in MODES:
            if s[m]:
                print(f"   successive {m:<8} L{lo}: resid norm ratio {s[m]['resid_ratio'][lo]:.4f} cos {s[m]['resid_cos'][lo]:.3f}; "
                      f"block L{lo} param norm ratio {s[m]['param_ratio'][lo]:.4f} cos {s[m]['param_cos'][lo]:.3f}; "
                      f"band-mean param ratio {np.mean([s[m]['param_ratio'][l] for l in band]):.4f}")

    name = f"15_gradient_capture/{args.model}"
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for m in MODES:
        axes[0].plot(lens_layers, summary["target"]["capture_by_layer_k10"][m], color=COLORS[m], label=m)
    axes[0].axhline(10 / d, color=NEUTRAL, ls="--", lw=1, label="chance k/d")
    shade_band(axes[0], (band[0], band[-1]), None)
    axes[0].set_yscale("log")
    axes[0].set_title("(a) single gate, k=10: captured gradient energy", loc="left")
    axes[0].set_xlabel("layer")
    axes[0].legend(fontsize=7)
    for m in MODES:
        axes[1].plot(KS, [summary["target"]["band_capture"][m][k] for k in KS], marker="o", color=COLORS[m], label=m)
    axes[1].plot(KS, [k / d for k in KS], ls="--", color=NEUTRAL, lw=1, label="chance")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) band-mean capture vs k (target)", loc="left")
    axes[1].set_xlabel("k")
    for m in MODES:
        s = summary["target"]["successive"][m]
        axes[2].plot(range(L), s["param_ratio"], color=COLORS[m], label=m)
    shade_band(axes[2], (band[0], band[-1]), None)
    axes[2].set_yscale("log")
    axes[2].set_title(f"(c) successive gates (k={args.k_successive}): block weight-grad norm ratio", loc="left")
    axes[2].set_xlabel("block")
    fig.savefig(out_dir(name) / "gradient_capture.png")
    save_results(name, {"summary": summary, "ks": KS, "lens_layers": lens_layers}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
