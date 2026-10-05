"""Paper section 4.2 / Fig. 31: how many list items does the J-space hold?

The model reads an 80-word comma-separated list (raw text, ``"List: w1, w2,
..."``). At each comma, a list word counts as *present* if its best rank over
the workspace band (min over layers and over its single-token surface forms) is
below ``k`` (default 25). Conditions:

* ``family`` -- 80 words from one category canon (names / surnames /
  countries / cities; the first 100 single-token entries of each pool in
  ``data/experiments/capacity.json``).
* ``unrelated`` -- 80 random single-token lowercase English-like vocabulary
  words (no shared category).
* ``blocks`` -- four contiguous 20-word blocks, one per family, in shuffled
  order (eviction on category switch, paper Fig. 31E/F).

Reported per comma: words read so far that are present (solid line), and all
80 list words present including those not yet read (dashed line).
"""

from __future__ import annotations

import argparse
import random
import re

import numpy as np
import torch

from jspace.data import load_json
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, style
from jspace.readout import Readout
from jspace.tokens import WordSets


def canon(lm, cfg) -> dict[str, list[str]]:
    out = {}
    for p in cfg["candidate_pools"]:
        n = cfg["targets_per_family"][p["name"]]
        ok = [w for w in p["pool"] if len(lm.tok.encode(" " + w, add_special_tokens=False)) == 1]
        out[p["name"]] = ok[:n]
    return out


def random_words(lm, n_pool: int, rng: random.Random) -> list[str]:
    pat = re.compile(r"^ [a-z]{4,9}$")
    cands = []
    for i in range(min(lm.vocab_size, len(lm.tok))):
        s = lm.tok.decode([i])
        if pat.match(s):
            cands.append(s.strip())
    rng.shuffle(cands)
    return cands[:n_pool]


@torch.no_grad()
def presence(lm, words: list[str], band: list[int], k: int) -> tuple[np.ndarray, np.ndarray]:
    """``present[c, w]``: word ``w`` present (band-min rank < k) at comma ``c``."""
    text = "List: " + ", ".join(words) + ","
    ro = Readout(lm, lm.run(text))
    ids = ro.fw.ids.tolist()
    comma = [i for i, t in enumerate(ids) if lm.tok.decode([t]) == ","]
    assert len(comma) == len(words), (len(comma), len(words))
    ws = WordSets(lm.tok, words)
    assert len(ws) == len(words), ws.missing
    r = ro.word_ranks(ws, layers=band, positions=comma)  # [L, C, W]
    best = r.min(0).values.numpy()  # [C, W]
    return best < k, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--list-len", type=int, default=80)
    ap.add_argument("--trials", type=int, default=6, help="per family / unrelated / blocks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    cfg = load_json("experiments/capacity.json")
    fams = canon(lm, cfg)
    N = args.list_len
    res = {"family": {}, "unrelated": None, "blocks": None}

    def curves(present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C = present.shape[0]
        read = np.array([present[c, : c + 1].sum() for c in range(C)])
        allw = present.sum(1)
        return read, allw

    fam_read, fam_all = [], []
    for fam, words in fams.items():
        rs, als = [], []
        for _ in range(args.trials):
            lst = rng.sample(words, N)
            p, _ = presence(lm, lst, band, args.k)
            r, a = curves(p)
            rs.append(r)
            als.append(a)
        res["family"][fam] = {"read": np.mean(rs, 0).tolist(), "all": np.mean(als, 0).tolist()}
        fam_read += rs
        fam_all += als
        print(f"{fam:10s} read-so-far present @comma 1/8/40/80: {np.mean(rs,0)[[0,7,39,N-1]].round(1)}  all-80: {np.mean(als,0)[[0,7,39,N-1]].round(1)}")

    pool = random_words(lm, 4000, rng)
    un_read, un_all = [], []
    for _ in range(args.trials * 2):
        lst = rng.sample(pool, N)
        p, _ = presence(lm, lst, band, args.k)
        r, a = curves(p)
        un_read.append(r)
        un_all.append(a)
    res["unrelated"] = {"read": np.mean(un_read, 0).tolist(), "all": np.mean(un_all, 0).tolist(),
                        "read_iqr": np.percentile(un_read, [25, 75], axis=0).tolist()}
    print(f"unrelated  read-so-far present @comma 1/8/40/80: {np.mean(un_read,0)[[0,7,39,N-1]].round(1)}  all-80: {np.mean(un_all,0)[[0,7,39,N-1]].round(1)}")
    res["family_pooled"] = {"read": np.mean(fam_read, 0).tolist(), "all": np.mean(fam_all, 0).tolist(),
                            "read_iqr": np.percentile(fam_read, [25, 75], axis=0).tolist()}

    # block-switch lists: P(word present) per (word slot, comma), averaged over trials
    B = N // 4
    block_mat = np.zeros((N, N))
    order_log = []
    for _ in range(args.trials * 2):
        order = list(fams)
        rng.shuffle(order)
        order_log.append(order)
        lst = [w for f in order for w in rng.sample(fams[f], B)]
        p, _ = presence(lm, lst, band, args.k)
        block_mat += p.T  # [W, C]
    block_mat /= args.trials * 2
    # mean presence of a block's words: during its block vs. 5 commas after the switch
    during, after = [], []
    for b in range(3):
        rows = slice(b * B, (b + 1) * B)
        during.append(block_mat[rows, b * B + B // 2 : (b + 1) * B].mean())
        after.append(block_mat[rows, (b + 1) * B + 4 : (b + 1) * B + 8].mean())
    res["blocks"] = {"matrix": block_mat.tolist(), "orders": order_log,
                     "mean_presence_during_own_block": float(np.mean(during)),
                     "mean_presence_5_commas_after_switch": float(np.mean(after))}
    print(f"blocks: own-block presence {np.mean(during):.2f} -> {np.mean(after):.2f} five commas after the category switch")

    name = f"10_capacity/{args.model}"
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    x = np.arange(1, N + 1)
    for key, c, lab in (("family_pooled", SERIES[1], "single-family lists"), ("unrelated", SERIES[0], "unrelated words")):
        ax.plot(x, res[key]["read"], color=c, label=f"{lab}: read so far")
        ax.plot(x, res[key]["all"], color=c, ls="--", lw=1.2, label=f"{lab}: all 80")
        lo, hi = np.array(res[key]["read_iqr"])
        ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
    ax.set_xlabel("comma (list position)")
    ax.set_ylabel(f"list words in J-lens top-{args.k}")
    ax.legend(fontsize=7)
    ax = axes[1]
    im = ax.imshow(block_mat, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper")
    ax.grid(False)
    for b in range(1, 4):
        ax.axvline(b * B - 0.5, color=NEUTRAL, lw=0.6)
        ax.axhline(b * B - 0.5, color=NEUTRAL, lw=0.6)
    ax.set_xlabel("comma")
    ax.set_ylabel("list word (4 blocks of 20)")
    ax.set_title(f"P(word in top-{args.k}) in block-switch lists", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_dir(name) / "capacity.png")
    save_results(name, res, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
