"""Paper appendix A.6 / Fig. 52: does the lens surface known unspoken intermediates?

For each item of the six released eval sets, read the lens at the specified
position and record each intermediate's best rank over layers (min over its
single-token surface forms). pass@k = fraction of intermediates with best rank
< k; summarised by the normalised AUC of pass@k over log k, k in [1, 1000].
Compared: J-lens vs logit lens, over all layers and over the workspace band.
"""

from __future__ import annotations

import argparse

import numpy as np

from jspace.data import load_json
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, style
from jspace.readout import Readout
from jspace.tokens import WordSets

EVALS = ["multihop", "multilingual", "order-ops", "poetry", "typo", "association"]
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
OPS = {
    "multiplication": ["multiplication", "multiply", "times", "*", "×", "product"],
    "addition": ["addition", "add", "plus", "+", "sum"],
    "subtraction": ["subtraction", "subtract", "minus", "-", "difference"],
    "division": ["division", "divide", "divided", "/", "÷", "quotient"],
}
KS = np.unique(np.logspace(0, 3, 40).astype(int))


def synonyms(name: str, word: str) -> list[str]:
    if name != "order-ops":
        return [word]
    if word in OPS:
        return OPS[word]
    if word.isdigit() and int(word) < len(NUMBER_WORDS):
        return [word, NUMBER_WORDS[int(word)]]
    return [word]


def readout_position(lm, name: str, prompt: str) -> int:
    if name != "poetry":
        return -1
    ids = lm.encode(prompt).tolist()
    nl = [i for i, t in enumerate(ids) if "\n" in lm.tok.decode([t])]
    return nl[-1] - len(ids)


def auc(best_ranks: np.ndarray) -> float:
    curve = np.array([(best_ranks < k).mean() for k in KS])
    return float(np.trapezoid(curve, np.log(KS)) / np.log(KS[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    all_layers = lm.lens_layers
    results, examples = {}, {}
    for name in EVALS:
        items = load_json(f"evaluations/lens-eval-{name}.json")["items"]
        best = {(m, r): [] for m in ("jacobian", "logit") for r in ("all", "band")}
        argbest = []
        skipped = 0
        for it in items:
            pos = readout_position(lm, name, it["prompt"])
            fw = lm.run(it["prompt"])
            for inter in it["intermediates"]:
                ws = WordSets(lm.tok, synonyms(name, inter))
                if not len(ws):
                    skipped += 1
                    continue
                for m in ("jacobian", "logit"):
                    ro = Readout(lm, fw, method=m)
                    r = ro.word_ranks(ws, layers=all_layers, positions=[pos])[:, 0, :].min(-1).values
                    best[(m, "all")].append(int(r.min()))
                    best[(m, "band")].append(int(r[band].min()))
                    if m == "jacobian":
                        argbest.append((it["name"], inter, int(r.min()), int(r.argmin())))
        results[name] = {
            f"{m}_{r}": {"auc": auc(np.array(v)), "pass@1": float(np.mean(np.array(v) < 1)),
                         "pass@10": float(np.mean(np.array(v) < 10)), "n": len(v)}
            for (m, r), v in best.items()
        }
        results[name]["skipped_multitoken"] = skipped
        examples[name] = sorted(argbest, key=lambda x: x[2])[:8]
        print(name, {k: round(v["auc"], 3) for k, v in results[name].items() if isinstance(v, dict)},
              "p@10 J/LL", round(results[name]["jacobian_all"]["pass@10"], 2), round(results[name]["logit_all"]["pass@10"], 2),
              "skipped", skipped)
    name = f"02_lens_quality/{args.model}"
    style()
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(EVALS))
    for i, (m, c, lab) in enumerate((("jacobian", SERIES[0], "J-lens"), ("logit", NEUTRAL, "logit lens"))):
        ax.bar(x + (i - 0.5) * 0.38, [results[e][f"{m}_all"]["auc"] for e in EVALS], 0.36, color=c, label=lab)
    ax.set_xticks(x, EVALS)
    ax.set_ylabel("pass@k AUC (normalised)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(out_dir(name) / "lens_quality.png")
    save_results(name, {"results": results, "best_examples": examples}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
