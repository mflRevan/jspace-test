"""Paper section 4.2 / Fig. 30: J-space occupancy and share of variance.

Held-out residuals (mean-centred per layer over the sample) are decomposed by
nonnegative gradient pursuit onto (i) the full unit J-lens dictionary (one atom
per vocabulary token) and (ii) a same-size control dictionary: ``--control
random`` (isotropic random unit atoms, as in the paper's text) or ``--control
rotated`` (the same J-lens atoms under a fixed random orthogonal rotation,
preserving their spectrum and pairwise geometry -- the paper's ``J_rot``
control from section 4.3.2). Qwen3.5's J-lens atoms are strongly anisotropic,
so the isotropic control is a much stronger generic approximator. Occupancy at
a position is the first ``K`` at which the J-lens dictionary's marginal gain in
fraction of variance explained (FVE) falls below the random dictionary's. We
also report the excess FVE (J minus random) at ``K`` = median occupancy.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from jspace.data import heldout_passages
from jspace.decomposition import nn_gradient_pursuit, random_dictionary
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, shade_band, style


def occupancy(fve_j: torch.Tensor, fve_r: torch.Tensor) -> torch.Tensor:
    """First K (1-based) where the J marginal gain drops below random's; K_max if never."""
    dj = torch.diff(fve_j, dim=1, prepend=torch.zeros_like(fve_j[:, :1]))
    dr = torch.diff(fve_r, dim=1, prepend=torch.zeros_like(fve_r[:, :1]))
    below = dj < dr
    K = fve_j.shape[1]
    first = torch.where(below.any(1), below.float().argmax(1), torch.full_like(below[:, 0], K, dtype=torch.long))
    return first  # number of atoms that beat the random control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-passages", type=int, default=8)
    ap.add_argument("--n-pos", type=int, default=384)
    ap.add_argument("--k-max", type=int, default=60)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--control", choices=["random", "rotated"], default="rotated")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, args.lens)
    resid = torch.cat([lm.run(lm.encode(p)[:128]).resid[:, 16:].float() for p in heldout_passages(args.n_passages)], 1)
    g = torch.Generator().manual_seed(args.seed)
    sel = torch.randperm(resid.shape[1], generator=g)[: args.n_pos].to(resid.device)
    if args.control == "random":
        Drand = random_dictionary(lm.vocab_size, lm.d_model, device=lm.device, seed=args.seed)
    else:
        gr = torch.Generator(device=lm.device).manual_seed(args.seed)
        R = torch.linalg.qr(torch.randn(lm.d_model, lm.d_model, device=lm.device, generator=gr)).Q
    layers = list(range(0, lm.n_layers - 1, args.layer_stride))
    if lm.spec.band[1] not in layers:
        layers.append(lm.spec.band[1])
    out = {"layers": [], "occ_p25": [], "occ_median": [], "occ_p75": [], "fve_j": [], "fve_r": [], "excess_fve_at_median_occ": []}
    for l in sorted(layers):
        X = resid[l, sel]
        X = X - X.mean(0, keepdim=True)
        Dj = lm.lens_dictionary(l)
        dj = nn_gradient_pursuit(X, Dj, args.k_max)
        if args.control == "rotated":
            Drand = Dj
            for s0 in range(0, Dj.shape[0], 32768):
                Drand[s0 : s0 + 32768] = (Dj[s0 : s0 + 32768].float() @ R).to(Dj.dtype)
        else:
            del Dj
        dr = nn_gradient_pursuit(X, Drand, args.k_max)
        if args.control == "rotated":
            del Drand, Dj
        occ = occupancy(dj.fve, dr.fve).float().cpu().numpy()
        med = int(np.median(occ))
        kk = max(med, 1) - 1
        excess = float((dj.fve[:, kk] - dr.fve[:, kk]).median())
        out["layers"].append(l)
        out["occ_p25"].append(float(np.percentile(occ, 25)))
        out["occ_median"].append(float(med))
        out["occ_p75"].append(float(np.percentile(occ, 75)))
        out["fve_j"].append(dj.fve.median(0).values.cpu().tolist())
        out["fve_r"].append(dr.fve.median(0).values.cpu().tolist())
        out["excess_fve_at_median_occ"].append(excess)
        print(f"L{l:2d} occupancy median={med:3d} (IQR {np.percentile(occ,25):.0f}-{np.percentile(occ,75):.0f})  "
              f"FVE@K: J={dj.fve[:, kk].median():.3f} rand={dr.fve[:, kk].median():.3f} excess={excess:.3f}  "
              f"FVE@60 J={dj.fve[:, -1].median():.3f} rand={dr.fve[:, -1].median():.3f}")
        torch.cuda.empty_cache()

    name = f"13_occupancy/{args.model}/{args.control}"
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    x = np.array(out["layers"])
    ax = axes[0]
    ax.plot(x, out["occ_median"], color=SERIES[0], label="median")
    ax.fill_between(x, out["occ_p25"], out["occ_p75"], color=SERIES[0], alpha=0.15, lw=0, label="IQR")
    shade_band(ax, tuple(lm.spec.band))
    ax.set_xlabel("layer")
    ax.set_ylabel("occupancy K (J-lens atoms beating random)")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(x, [f[-1] for f in out["fve_j"]], color=SERIES[0], label=f"J-lens dictionary, K={args.k_max}")
    ax.plot(x, [f[-1] for f in out["fve_r"]], color=NEUTRAL, label=f"{args.control} control, K={args.k_max}")
    ax.plot(x, out["excess_fve_at_median_occ"], color=SERIES[1], label="excess FVE at K=median occ.")
    shade_band(ax, tuple(lm.spec.band), None)
    ax.set_xlabel("layer")
    ax.set_ylabel("fraction of variance explained")
    ax.legend(fontsize=7)
    fig.savefig(out_dir(name) / "occupancy.png")
    save_results(name, out, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
