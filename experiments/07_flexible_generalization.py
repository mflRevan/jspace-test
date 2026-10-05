"""Paper section 3.4 / Figs. 18-19: one J-lens vector is a valid argument to many
downstream functions.

For each category (countries, months, animals, numbers), each of 4 function
templates and each ordered pair of its 4 arguments (192 trials), the source
argument's lens coordinates are swapped with the target's at every band layer
and position. Success (strict) = greedy output begins with the target's answer;
success (set) = the target's answer has the highest log-prob among the four
answers of that function. Run at alpha = 1 and 2 (paper: 76/192 and 101/192).

Workspace loading of the source = cosine(residual, source J-lens vector),
averaged over band layers at the argument and final positions (clean run).
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from jspace.analysis import find_token
from jspace.data import load_json
from jspace.generate import continuation_logprobs, generate
from jspace.interventions import Swap
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import SERIES, plt, style
from jspace.tokens import paired_forms, primary_token_id


def matches(text: str, ans: str) -> bool:
    return text.strip().lower().lstrip("\"'*").startswith(ans.lower())


@torch.no_grad()
def loading(lm, fw, tid, positions, layers) -> float:
    vals = []
    for l in layers:
        v = lm.lens_vectors([tid], l, unit=True)[0]
        h = fw.resid[l, positions].float()
        vals.append((h @ v / h.norm(dim=-1)).mean().item())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    cats = load_json("experiments/flexible-generalization.json")["categories"]
    trials = []
    for cat in cats:
        for fn in cat["funcs"]:
            answers = fn["answers"]
            conts = [" " + answers[a] for a in cat["args"]]
            for src, tgt in itertools.permutations(cat["args"], 2):
                prompt = fn["template"].format(arg=src)
                ids = lm.encode(prompt)
                fw = lm.run(ids)
                clean = generate(lm, ids, max_new_tokens=5).text
                s_ids, t_ids = paired_forms(lm.tok, src, tgt)
                tid = primary_token_id(lm.tok, src)
                apos = find_token(lm, prompt, src)
                rec = {"category": cat["name"], "func": fn["name"], "src": src, "tgt": tgt,
                       "clean": clean, "clean_correct": matches(clean, answers[src]),
                       "loading": loading(lm, fw, tid, [apos, -1], band) if tid is not None else None}
                for a in args.alphas:
                    e = Swap(layers=band, src=s_ids, tgt=t_ids, alpha=a)
                    lp = continuation_logprobs(lm, ids, conts, [e])
                    text = generate(lm, ids, [e], max_new_tokens=5).text
                    rec[f"a{a:g}"] = {"text": text, "strict": matches(text, answers[tgt]),
                                      "set": int(np.argmax(lp)) == cat["args"].index(tgt),
                                      "degenerate": matches(text, tgt) and not matches(answers[tgt], tgt)}
                trials.append(rec)
            ok = [t for t in trials if t["func"] == fn["name"] and t["category"] == cat["name"]]
            print(f"{cat['name']:<9} {fn['name']:<12} clean-correct {sum(t['clean_correct'] for t in ok)}/12  "
                  + "  ".join(f"a={a:g}: strict {sum(t[f'a{a:g}']['strict'] for t in ok)}/12 set {sum(t[f'a{a:g}']['set'] for t in ok)}/12"
                              for a in args.alphas)
                  + f"   e.g. {ok[0]['src']}->{ok[0]['tgt']}: {ok[0]['clean'].strip()[:12]!r} -> {ok[0]['a1']['text'].strip()[:14]!r}")

    summary = {}
    for a in args.alphas:
        k = f"a{a:g}"
        summary[k] = {"strict": sum(t[k]["strict"] for t in trials), "set": sum(t[k]["set"] for t in trials),
                      "degenerate": sum(t[k]["degenerate"] for t in trials), "n": len(trials),
                      "by_category": {c["name"]: sum(t[k]["strict"] for t in trials if t["category"] == c["name"]) for c in cats}}
    lo = np.array([t["loading"] if t["loading"] is not None else np.nan for t in trials])
    succ = np.array([t["a1"]["set"] for t in trials], dtype=float)
    m = ~np.isnan(lo)
    summary["loading_vs_success_corr"] = float(np.corrcoef(lo[m], succ[m])[0, 1])
    summary["mean_loading_by_category"] = {c["name"]: float(np.nanmean([t["loading"] or np.nan for t in trials if t["category"] == c["name"]])) for c in cats}
    print(summary)
    name = f"07_flexible_generalization/{args.model}"
    style()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, c in enumerate(cats):
        sel = [t for t in trials if t["category"] == c["name"] and t["loading"] is not None]
        ax.scatter([t["loading"] for t in sel], [t["a1"]["set"] + np.random.default_rng(i).uniform(-.08, .08) for t in sel],
                   s=12, color=SERIES[i], label=c["name"], alpha=0.7)
    ax.set_xlabel("workspace loading of source (cosine)")
    ax.set_ylabel("swap success (alpha=1, jittered)")
    ax.legend()
    fig.savefig(out_dir(name) / "flexible_generalization.png")
    save_results(name, {"summary": summary, "trials": trials}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
