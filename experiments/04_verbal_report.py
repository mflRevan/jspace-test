"""Paper section 3.1 / Fig. 6: the J-space determines verbal report.

"Think of a {category}. Answer in one word." (14 categories x 14 candidates,
data/experiments/verbal-report.json). At the assistant-start position:

(a) Correlation: Spearman between each candidate's J-lens logit at a band layer
    (best over its surface forms) and its output log-prob (whole-string score).
(b) Swap: the lens coordinates of the model's own answer are exchanged with a
    candidate's at every band layer and every position (alpha = 1); candidates
    already in the model's top-10 are skipped. Success = the greedy answer
    becomes the candidate (strict), or the candidate becomes the argmax of the
    candidate set (lenient). Output rank of each target before/after is kept.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from jspace.data import load_json
from jspace.generate import continuation_logprobs, generate
from jspace.interventions import Swap
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, style
from jspace.readout import Readout
from jspace.tokens import WordSets, paired_forms, single_token_ids


def first_word(text: str) -> str:
    w = text.strip().split()
    return w[0].strip(".,!*\"'") if w else ""


def swap_ids(tok, a: str, b: str) -> tuple[list[int], list[int]]:
    s, t = paired_forms(tok, a, b)
    if s:
        return s, t
    # multi-token words: fall back to first tokens of the capitalised forms
    sa = single_token_ids(tok, a.capitalize(), allow_prefix=True)[:1]
    tb = single_token_ids(tok, b.capitalize(), allow_prefix=True)[:1]
    return (sa, tb) if sa and tb and sa != tb else ([], [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--max-targets", type=int, default=10)
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    corr_layers = [band[0], band[len(band) // 2], band[-1]]
    cands_all = load_json("experiments/verbal-report.json")["candidates"]

    corr = {l: [] for l in corr_layers}
    trials, examples = [], []
    for cat, cands in cands_all.items():
        prompt = lm.chat(f"Think of a {cat}. Answer in one word.")
        ids = lm.encode(prompt)
        ro = Readout(lm, lm.run(ids))
        clean_text = generate(lm, ids, max_new_tokens=6).text
        answer = first_word(clean_text)
        conts = [c.capitalize() for c in cands]
        out_lp = np.array(continuation_logprobs(lm, ids, conts))
        ws = WordSets(lm.tok, cands, allow_prefix=True)
        lens_r = ro.word_ranks(ws, layers=corr_layers, positions=[-1])[:, 0]  # [3, n]
        idx = [cands.index(w) for w in ws.words]
        for li, l in enumerate(corr_layers):
            rho = spearmanr(-lens_r[li].numpy(), out_lp[idx]).statistic
            corr[l].append(float(rho))
        order = np.argsort(-out_lp)
        top10 = {cands[i].lower() for i in order[:10]}
        targets = [c for c in cands if c.lower() not in top10 and c.lower() != answer.lower()]
        for tgt in targets[: args.max_targets]:
            src_ids, tgt_ids = swap_ids(lm.tok, answer, tgt)
            if not src_ids:
                continue
            e = Swap(layers=band, src=src_ids, tgt=tgt_ids, alpha=1.0)
            sw_lp = np.array(continuation_logprobs(lm, ids, conts, [e]))
            sw_text = generate(lm, ids, [e], max_new_tokens=6).text
            ti = cands.index(tgt)
            rank_before = int((out_lp > out_lp[ti]).sum())
            rank_after = int((sw_lp > sw_lp[ti]).sum())
            rec = {
                "category": cat, "answer": answer, "target": tgt,
                "rank_before": rank_before, "rank_after": rank_after,
                "strict": first_word(sw_text).lower() == tgt.lower(),
                "lenient": rank_after == 0, "swapped_text": sw_text,
                "n_form_pairs": len(src_ids),
            }
            trials.append(rec)
        ex = [t for t in trials if t["category"] == cat]
        print(f"{cat:<11} answer={answer!r:<14} lens@L{band[-1]} top: "
              f"{[t for t, _ in ro.topk(band[-1], -1, 5)]}  swaps strict "
              f"{sum(t['strict'] for t in ex)}/{len(ex)}  e.g. {[(t['target'], t['swapped_text'].strip()[:14]) for t in ex[:3]]}")
        examples.append({"category": cat, "clean": clean_text,
                         "lens_top_last_band_layer": ro.topk(band[-1], -1, 10)})

    n = len(trials)
    summary = {
        "spearman_median": {l: float(np.nanmedian(v)) for l, v in corr.items()},
        "swap_strict": sum(t["strict"] for t in trials) / n,
        "swap_lenient": sum(t["lenient"] for t in trials) / n,
        "swap_top5": sum(t["rank_after"] < 5 for t in trials) / n,
        "n_trials": n,
    }
    print(summary)
    name = f"04_verbal_report/{args.model}"
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].boxplot([corr[l] for l in corr_layers], tick_labels=[f"L{l}" for l in corr_layers])
    axes[0].set_title("Spearman: J-lens vs output (14 categories)", loc="left")
    rb = np.array([t["rank_before"] for t in trials])
    ra = np.array([t["rank_after"] for t in trials])
    jit = np.random.default_rng(0).uniform(-0.25, 0.25, size=n)
    axes[1].scatter(rb + jit, ra + jit[::-1], s=10, color=SERIES[0], alpha=0.6)
    axes[1].plot([0, 13], [0, 13], color=NEUTRAL, lw=1)
    axes[1].set_xlabel("target rank before swap (of 14)")
    axes[1].set_ylabel("rank after swap")
    axes[1].set_title("Swap moves the target to the top", loc="left")
    fig.savefig(out_dir(name) / "verbal_report.png")
    save_results(name, {"summary": summary, "trials": trials, "examples": examples,
                        "spearman": corr}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
