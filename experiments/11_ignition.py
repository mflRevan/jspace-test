"""Paper section 4.1.1 / Fig. 29: interpretation of ambiguous inputs solidifies
at the workspace onset ("ignition").

A country token's input embedding is replaced by ``(1 - a) e_B + a e_A`` in a
carrier sentence; ``a`` sweeps 0 -> 1 in 21 steps. At the mixed position and
every layer we record:

* ``proj`` -- projection share of ``h(a)`` along the line from the pure-B
  (``a=0``) to the pure-A (``a=1``) activation (0 at B, 1 at A by construction);
* ``jproj`` -- the same, restricted to the activation's coordinates on the two
  concepts' unit J-lens vectors (pseudoinverse coordinates);
* ``rrshare`` -- J-lens reciprocal-rank share ``rr(A) / (rr(A) + rr(B))``.

Each trial's threshold is the ``a`` at which the final-layer ``proj`` crosses
0.5. Per layer we report the 10->90% transition width in ``a`` (median over
trials; small = sharp). For bimodality, each concept pair gets one maximally
ambiguous mixture (the median of its trials' thresholds); across carrier
sentences at that fixed ``a`` we report the fraction of trials in the ambiguous
middle [0.25, 0.75] (small = bimodal, all-or-none).
"""

from __future__ import annotations

import argparse
import itertools
import random

import numpy as np
import torch
from jlens.hooks import ActivationRecorder

from jspace.data import load_json
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import SERIES, plt, shade_band, style


def crossing(alphas: np.ndarray, y: np.ndarray, level: float) -> float:
    """First ``a`` where ``y`` (roughly increasing in ``a``) reaches ``level``."""
    idx = np.nonzero(y >= level)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(alphas[0])
    y0, y1 = y[i - 1], y[i]
    return float(alphas[i - 1] + (level - y0) / max(y1 - y0, 1e-9) * (alphas[i] - alphas[i - 1]))


@torch.no_grad()
def run_trial(lm, template: str, A: int, B: int, alphas: torch.Tensor):
    prefix = template.split("{W}")[0].rstrip()
    suffix = template.split("{W}")[1]
    pre = lm.tok.encode(prefix, add_special_tokens=True)
    suf = lm.tok.encode(suffix, add_special_tokens=False) if suffix.strip() else []
    ids = torch.tensor(pre + [A] + suf, device=lm.device)
    pos = len(pre)
    E = lm.hf.model.embed_tokens(ids)[None].repeat(len(alphas), 1, 1)  # [n_a, T, d]
    eA, eB = lm.hf.model.embed_tokens.weight[A], lm.hf.model.embed_tokens.weight[B]
    E[:, pos] = ((1 - alphas)[:, None] * eB.float() + alphas[:, None] * eA.float()).to(E.dtype)
    with ActivationRecorder(lm.layers, at=range(lm.n_layers)) as rec:
        lm.hf.model(inputs_embeds=E, use_cache=False)
        H = torch.stack([rec.activations[l][:, pos].float() for l in range(lm.n_layers)])  # [L, n_a, d]
    hB, hA = H[:, :1], H[:, -1:]
    line = hA - hB
    proj = ((H - hB) * line).sum(-1) / line.pow(2).sum(-1).clamp_min(1e-9)  # [L, n_a]
    jproj = torch.full_like(proj, float("nan"))
    rrshare = torch.full_like(proj, float("nan"))
    for l in range(lm.n_layers):
        V = lm.lens_vectors([A, B], l, unit=True).T  # [d, 2]
        c = H[l] @ torch.linalg.pinv(V).T  # [n_a, 2]
        cl = c[-1] - c[0]
        jproj[l] = ((c - c[0]) * cl).sum(-1) / cl.pow(2).sum().clamp_min(1e-9)
        lg = lm.lens_logits(H[l], l)
        tgt = lg[:, [A, B]]
        rk = (lg[:, None, :] > tgt[:, :, None]).sum(-1).float() + 1  # [n_a, 2]
        rr = 1 / rk
        rrshare[l] = rr[:, 0] / rr.sum(-1)
    return proj.cpu().numpy(), jproj.cpu().numpy(), rrshare.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--n-templates", type=int, default=20)
    ap.add_argument("--steps", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    lm = LensedModel(args.model, args.lens)
    cfg = load_json("experiments/ignition.json")
    single = {c: lm.tok.encode(" " + c, add_special_tokens=False) for c in cfg["countries_12"]}
    single = {c: t[0] for c, t in single.items() if len(t) == 1}
    pairs = list(itertools.combinations(sorted(single), 2))
    rng.shuffle(pairs)
    pairs = pairs[: args.n_pairs]
    templates = cfg["ctx_templates"][: args.n_templates]
    alphas = torch.linspace(0, 1, args.steps, device=lm.device)
    a_np = alphas.cpu().numpy()
    L = lm.n_layers

    widths = {k: [] for k in ("proj", "jproj", "rrshare")}
    at_thr = {k: [] for k in widths}
    aligned = {k: [] for k in widths}  # curves re-indexed relative to threshold
    rel_grid = np.linspace(-0.5, 0.5, 41)
    trials = []  # (pair index, threshold, vals)
    for pi, ((ca, cb), tpl) in enumerate(itertools.product(pairs, templates)):
        vals = dict(zip(("proj", "jproj", "rrshare"), run_trial(lm, tpl, single[ca], single[cb], alphas), strict=True))
        thr = crossing(a_np, vals["proj"][-1], 0.5)
        if np.isnan(thr):
            continue
        trials.append((pi // len(templates), thr, vals))
        for k, v in vals.items():
            w = [crossing(a_np, v[l], 0.9) - crossing(a_np, v[l], 0.1) for l in range(L)]
            widths[k].append(w)
            aligned[k].append(np.stack([np.interp(rel_grid + thr, a_np, v[l], left=np.nan, right=np.nan) for l in range(L)]))
    pair_thr = {p: np.median([t for q, t, _ in trials if q == p]) for p in {q for q, _, _ in trials}}
    for p, _, vals in trials:
        ti = int(np.argmin(np.abs(a_np - pair_thr[p])))
        for k, v in vals.items():
            at_thr[k].append(v[:, ti])
    n = len(trials)
    summary = {}
    for k in widths:
        W = np.array(widths[k])
        T = np.array(at_thr[k])
        summary[k] = {
            "median_width": np.nanmedian(W, 0).tolist(),
            "frac_middle_at_threshold": np.nanmean((T > 0.25) & (T < 0.75), 0).tolist(),
            "aligned_mean": np.nanmean(np.array(aligned[k]), 0).tolist(),
        }
    for l in range(0, L, 2):
        print(f"L{l:2d} width proj={summary['proj']['median_width'][l]:.2f} jproj={summary['jproj']['median_width'][l]:.2f} "
              f"| middle@thr proj={summary['proj']['frac_middle_at_threshold'][l]:.2f} jproj={summary['jproj']['frac_middle_at_threshold'][l]:.2f} "
              f"rrshare={summary['rrshare']['frac_middle_at_threshold'][l]:.2f}")

    name = f"11_ignition/{args.model}"
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for ax, (k, title) in zip(axes[:2], (("proj", "projection share (full activation)"), ("jproj", "projection share (J-lens coords of A,B)")), strict=True):
        im = ax.imshow(np.array(summary[k]["aligned_mean"]), aspect="auto", origin="lower", cmap="RdBu_r", vmin=0, vmax=1,
                       extent=[rel_grid[0], rel_grid[-1], -0.5, L - 0.5])
        ax.grid(False)
        ax.axhspan(lm.spec.band[0] - 0.5, lm.spec.band[1] + 0.5, fill=False, ec="k", lw=0.6)
        ax.set_xlabel("a - threshold")
        ax.set_ylabel("layer")
        ax.set_title(title, loc="left")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    fig.subplots_adjust(wspace=0.38)
    ax = axes[2]
    x = np.arange(L)
    for k, c, lab in (("proj", SERIES[0], "full activation"), ("jproj", SERIES[1], "J-lens coords"), ("rrshare", SERIES[2], "J-lens rank share")):
        ax.plot(x, summary[k]["frac_middle_at_threshold"], color=c, label=lab)
    shade_band(ax, tuple(lm.spec.band))
    ax.set_xlabel("layer")
    ax.set_ylabel("frac. ambiguous at pair's threshold a")
    ax.set_title("lower = more bimodal (all-or-none)", loc="left")
    ax.legend(fontsize=7)
    fig.savefig(out_dir(name) / "ignition.png")
    save_results(name, {"pairs": pairs, "templates": templates, "n_trials": n, "alphas": a_np.tolist(),
                        "rel_grid": rel_grid.tolist(), "summary": summary}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
