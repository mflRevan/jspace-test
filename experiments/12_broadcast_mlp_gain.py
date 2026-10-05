"""Paper section 4.3.1 / Fig. 32: do MLPs preferentially amplify J-space directions?

For each source layer ``l`` and a population of unit directions ``v`` at that
layer, the MLP gain is ``|| mlp_{l+1}( post_attention_layernorm_{l+1}(s * v) ) ||``
(``s`` = mean residual norm at layer ``l``; the RMSNorm makes the result
nearly scale-free), normalised by the median gain of isotropic random unit
directions. Populations: J-lens vectors of random word-like tokens, output
directions of layer-``l`` MLP neurons (columns of ``down_proj``), and random
directions (control, gain ~1 by construction).

Two variants are reported:

* ``isolated`` -- the literal definition above (the MLP sees ``v`` alone).
* ``in_context`` -- the linearised gain at real activations: for held-out
  residuals ``h`` at layer ``l``, ``|| d/de mlp(norm(h + e v)) ||`` at ``e=0``
  (a Jacobian-vector product), averaged over positions, again normalised by the
  random-direction median. The paper notes (footnote 5) that gain in practice is
  context-dependent; this variant measures it in context.
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


@torch.no_grad()
def mlp_gain(block, dirs: torch.Tensor, scale: float, batch: int = 512) -> torch.Tensor:
    """``||mlp(norm(scale * v))||`` for unit rows of ``dirs`` ``[n, d]``."""
    out = []
    dtype = block.mlp.down_proj.weight.dtype
    for s in range(0, dirs.shape[0], batch):
        x = (scale * dirs[s : s + batch]).to(dtype)[None]
        y = block.mlp(block.post_attention_layernorm(x))[0]
        out.append(y.float().norm(dim=-1))
    return torch.cat(out)


def in_context_gain(block, H: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Mean over rows of ``H`` ``[P, d]`` of ``||J_mlp(h) v||`` for unit ``dirs`` ``[n, d]`` (fp32)."""
    norm = block.post_attention_layernorm
    mlp = block.mlp

    def f(x):
        return mlp(norm(x))

    acc = torch.zeros(dirs.shape[0], device=dirs.device)
    for h in H:
        x = h[None].expand_as(dirs).contiguous()
        _, jv = torch.func.jvp(f, (x,), (dirs,))
        acc += jv.norm(dim=-1)
    return acc / H.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-context", type=int, default=500, help="directions per population (in-context)")
    ap.add_argument("--n-pos", type=int, default=16, help="held-out positions per layer (in-context)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, args.lens)
    g = torch.Generator().manual_seed(args.seed)
    cand = wordlike_mask(lm.tok, lm.vocab_size).nonzero()[:, 0]
    tok_ids = cand[torch.randperm(len(cand), generator=g)[: args.n]].tolist()

    # typical residual norm per layer from a few held-out passages
    norms = np.zeros(lm.n_layers)
    passages = heldout_passages(8)
    pool = []
    for p in passages:
        fw = lm.run(lm.encode(p)[:128])
        norms += fw.resid[:, 16:].float().norm(dim=-1).mean(-1).cpu().numpy()
        pool.append(fw.resid[:, 16:].float())
    norms /= len(passages)
    pool = torch.cat(pool, dim=1)  # [n_layers, N, d]
    pos_idx = torch.randperm(pool.shape[1], generator=g)[: args.n_pos]

    layers = list(range(lm.n_layers - 1))  # source l, MLP of block l+1
    med = {"jlens": [], "neuron": [], "random": []}
    q = {k: [] for k in med}
    ctx = {k: [] for k in med}
    for l in layers:
        block = lm.layers[l + 1]
        d = lm.d_model
        rnd = torch.randn(args.n, d, generator=g).to(lm.device)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True)
        base = mlp_gain(block, rnd, norms[l]).median()
        jl = lm.lens_vectors(tok_ids, l, unit=True)
        W = lm.layers[l].mlp.down_proj.weight.float()  # [d, ffn]
        cols = torch.randperm(W.shape[1], generator=g)[: args.n].to(lm.device)
        neu = W[:, cols].T
        neu = neu / neu.norm(dim=-1, keepdim=True)
        for k, dirs in (("jlens", jl), ("neuron", neu), ("random", rnd)):
            gains = (mlp_gain(block, dirs, norms[l]) / base).cpu().numpy()
            med[k].append(float(np.median(gains)))
            q[k].append([float(np.percentile(gains, 25)), float(np.percentile(gains, 75))])
        # in-context linearised gain on fp32 copies of the block's norm + MLP
        import copy

        blk32 = copy.deepcopy(block).float()
        H = pool[l, pos_idx.to(pool.device)]
        nc = args.n_context
        cbase = in_context_gain(blk32, H, rnd[:nc]).median()
        for k, dirs in (("jlens", jl), ("neuron", neu), ("random", rnd)):
            ctx[k].append(float((in_context_gain(blk32, H, dirs[:nc]) / cbase).median()))
        del blk32
        print(f"L{l:2d}->{l+1:2d}  isolated J={med['jlens'][-1]:5.2f} neuron={med['neuron'][-1]:5.2f} | "
              f"in-context J={ctx['jlens'][-1]:5.2f} neuron={ctx['neuron'][-1]:5.2f}")

    name = f"12_broadcast_mlp_gain/{args.model}"
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    x = np.array(layers)
    pops = (("jlens", SERIES[0], "J-lens vectors"), ("neuron", SERIES[1], "MLP neuron output dirs"), ("random", NEUTRAL, "random"))
    for ax, (vals, title) in zip(axes, ((med, "isolated direction"), (ctx, "in context (linearised at real h)")), strict=True):
        for k, c, lab in pops:
            ax.plot(x, vals[k], color=c, label=lab)
        if vals is med:
            for k, c, _ in pops:
                lo, hi = np.array(q[k]).T
                ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
        shade_band(ax, tuple(lm.spec.band))
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("source layer l (MLP of block l+1)")
    axes[0].set_ylabel("MLP gain (x random median)")
    axes[0].legend(loc="upper left")
    fig.savefig(out_dir(name) / "mlp_gain.png")
    save_results(name, {"layers": layers, "median_gain_isolated": med, "iqr_isolated": q,
                  "median_gain_in_context": ctx, "resid_norm": norms.tolist()},
                 run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
