"""Paper section 4.1 / Fig. 27-28: where does the J-space act as a workspace?

Per layer, on held-out WikiText (positions >= 16, 128-token passages):
  (a) next-token accuracy: model's top-1 token within the lens top-1 / top-10
  (b) excess kurtosis of the lens logit distribution (readout peakedness)
  (c) top-1 autocorrelation: log P(top1[t] == top1[t+d]) minus a
      position-shuffled null, averaged over d = 1..4
  (d) effective dimensionality of the J-lens vectors (fraction of residual
      dims for 50% / 90% of variance)
  (e) linear CKA between layers of the J-lens cosine-similarity structure
(a)-(c) are reported for the J-lens and the logit lens.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from jspace.data import heldout_passages
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, shade_band, style
from jspace.tokens import wordlike_mask

SKIP = 16


def readout_stats(lm, passages, max_len):
    L = lm.n_layers
    acc1 = {m: np.zeros(L) for m in ("jacobian", "logit")}
    acc10 = {m: np.zeros(L) for m in ("jacobian", "logit")}
    kurt = {m: np.zeros(L) for m in ("jacobian", "logit")}
    auto = {m: np.zeros(L) for m in ("jacobian", "logit")}
    n_pos, n_pass = 0, 0
    g = torch.Generator().manual_seed(0)
    for text in passages:
        ids = lm.encode(text)[:max_len]
        fw = lm.run(ids)
        target = fw.logits[SKIP:-1].argmax(-1)
        P = target.shape[0]
        for method in ("jacobian", "logit"):
            for l in range(L):
                lg = lm.lens_logits(fw.resid[l, SKIP:-1], l, method=method)
                top10 = lg.topk(10, dim=-1).indices
                acc1[method][l] += (top10[:, 0] == target).sum().item()
                acc10[method][l] += (top10 == target[:, None]).any(-1).sum().item()
                z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
                kurt[method][l] += (z.pow(4).mean(-1) - 3).sum().item()
                t1 = top10[:, 0].cpu()
                same = np.mean([(t1[d:] == t1[:-d]).float().mean().item() for d in range(1, 5)])
                perm = t1[torch.randperm(P, generator=g)]
                null = np.mean([(perm[d:] == perm[:-d]).float().mean().item() for d in range(1, 5)])
                auto[method][l] += np.log((same + 1e-3) / (null + 1e-3))
        n_pos += P
        n_pass += 1
    return {
        "acc_top1": {m: (v / n_pos).tolist() for m, v in acc1.items()},
        "acc_top10": {m: (v / n_pos).tolist() for m, v in acc10.items()},
        "excess_kurtosis": {m: (v / n_pos).tolist() for m, v in kurt.items()},
        "autocorr_logratio": {m: (v / n_pass).tolist() for m, v in auto.items()},
        "n_positions": n_pos,
    }


@torch.no_grad()
def geometry_stats(lm, n_tokens, seed):
    mask = wordlike_mask(lm.tok, lm.vocab_size)
    cand = mask.nonzero()[:, 0]
    g = torch.Generator().manual_seed(seed)
    ids = cand[torch.randperm(len(cand), generator=g)[:n_tokens]].tolist()
    L, d = lm.n_layers, lm.d_model
    grams, eff50, eff90 = [], [], []
    for l in range(L):
        V = lm.lens_vectors(ids, l, unit=True)  # [n, d]
        Vc = V - V.mean(0, keepdim=True)
        s2 = torch.linalg.svdvals(Vc).pow(2)
        cum = (s2.cumsum(0) / s2.sum()).cpu().numpy()
        eff50.append(float((np.searchsorted(cum, 0.5) + 1) / d))
        eff90.append(float((np.searchsorted(cum, 0.9) + 1) / d))
        K = V @ V.T
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1.0 / n
        grams.append(H @ K @ H)
    cka = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            num = (grams[i] * grams[j]).sum()
            den = grams[i].norm() * grams[j].norm()
            cka[i, j] = cka[j, i] = float(num / den)
    return {"cka": cka.tolist(), "eff_dim_50": eff50, "eff_dim_90": eff90, "n_tokens": n_tokens}


def pick_band(stats):
    """Onset: first layer where J-lens autocorrelation reaches half its max.
    End: last layer before J-lens top-1 next-token accuracy exceeds half of its
    value at the final lens layer (the 'motor' transition)."""
    auto = np.array(stats["autocorr_logratio"]["jacobian"][:-1])
    acc = np.array(stats["acc_top1"]["jacobian"][:-1])
    onset = int(np.argmax(auto >= 0.5 * auto.max()))
    motor = int(np.argmax(acc >= 0.5 * acc[-1]))
    return onset, max(onset, motor - 1)


def plot(stats, geo, band, path, n_layers):
    style()
    x = np.arange(n_layers)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.2))
    panels = [
        ("acc_top10", "(a) next-token acc. (lens top-10)"),
        ("excess_kurtosis", "(b) excess kurtosis of readout"),
        ("autocorr_logratio", "(c) top-1 autocorrelation vs null (log)"),
    ]
    for ax, (k, title) in zip(axes[:3], panels, strict=True):
        for m, c, lab in (("jacobian", SERIES[0], "J-lens"), ("logit", NEUTRAL, "logit lens")):
            ax.plot(x, stats[k][m], color=c, label=lab)
        shade_band(ax, band)
        ax.set_title(title, loc="left")
        ax.set_xlabel("layer")
    axes[0].legend(loc="upper left")
    ax = axes[3]
    ax.plot(x, geo["eff_dim_50"], color=SERIES[0], label="50% var")
    ax.plot(x, geo["eff_dim_90"], color=SERIES[1], label="90% var")
    shade_band(ax, band, None)
    ax.set_title("(d) J-lens vectors: frac. dims", loc="left")
    ax.set_xlabel("layer")
    ax.legend(loc="lower right")
    ax = axes[4]
    im = ax.imshow(np.array(geo["cka"]), cmap="Blues", vmin=0, vmax=1, origin="lower")
    ax.grid(False)
    ax.set_title("(e) CKA of J-lens geometry", loc="left")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-passages", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--n-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, args.lens)
    stats = readout_stats(lm, heldout_passages(args.n_passages), args.max_len)
    geo = geometry_stats(lm, args.n_tokens, args.seed)
    band = pick_band(stats)
    name = f"01_layer_structure/{args.model}"
    plot(stats, geo, band, out_dir(name) / "layer_structure.png", lm.n_layers)
    save_results(name, {"stats": stats, "geometry": geo, "band": band}, run_meta(lm, args=vars(args)))
    print("band", band)
    for l in range(lm.n_layers):
        print(f"L{l:2d} acc10 J={stats['acc_top10']['jacobian'][l]:.2f} LL={stats['acc_top10']['logit'][l]:.2f} "
              f"kurt J={stats['excess_kurtosis']['jacobian'][l]:7.1f} LL={stats['excess_kurtosis']['logit'][l]:7.1f} "
              f"auto J={stats['autocorr_logratio']['jacobian'][l]:.2f} LL={stats['autocorr_logratio']['logit'][l]:.2f} "
              f"eff90={geo['eff_dim_90'][l]:.2f}")


if __name__ == "__main__":
    main()
