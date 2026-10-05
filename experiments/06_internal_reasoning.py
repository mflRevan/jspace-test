"""Paper section 3.3 / Figs. 12-17: J-lens vectors carry unspoken intermediates
and swapping them redirects the conclusion.

Parts
  A. Curated cases (lens evidence + clamped lens-coordinate swap at every band
     layer and position, alpha = 1): spider -> ant (legs), rhyme planning
     (fight -> light), Chinese antonym via English (big -> long), and a two-armed
     bandit "plan swap" (repeat <-> switch strategy sets).
  B. Arithmetic intermediates by layer: J-lens rank of each partial result at the
     final position (number words, since Qwen tokenizes digits singly).
  C. Systematic two-hop swaps (data/experiments/probe-swap.json, 90 items):
     swap intermediate -> swap_to; success = greedy answer becomes swap_answer
     (strict) or log P(swap_answer) > log P(answer) (pairwise).
  D. Depth test: swap the intermediate vs swap the answer within sliding
     4-layer windows; the intermediate swap should act at shallower depth.
"""

from __future__ import annotations

import argparse

import numpy as np

from jspace.analysis import band_top, find_token
from jspace.data import load_json
from jspace.generate import continuation_logprobs, generate
from jspace.interventions import SetSwap, Swap, lens_coords
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, shade_band, style
from jspace.readout import Readout
from jspace.tokens import WordSets, paired_forms, single_token_ids


def pair_ids(tok, pairs):
    src, tgt = [], []
    for a, b in pairs:
        s, t = paired_forms(tok, a, b)
        src += s
        tgt += t
    return src, tgt


def cont(prompt: str, ans: str) -> str:
    return ans if prompt.endswith((" ", '"', "\n")) else " " + ans


def matches(text: str, ans: str) -> bool:
    return text.strip().lower().lstrip("\"'*").startswith(ans.lower())


# --------------------------------------------------------------------- part A
def curated(lm, band):
    out = []
    cases = [
        dict(slug="spider-legs", prompt="Fact: The number of legs on the animal that spins webs is",
             pairs=[("spider", "ant"), ("spiders", "ants")], probe=" webs",
             answers=[" 8", " 6", " eight", " six"]),
        dict(slug="boot-currency", prompt="Fact: The currency used in the country shaped like a boot is the",
             pairs=[("Italy", "Japan"), ("Italian", "Japanese")], probe=" boot",
             answers=[" euro", " yen", " Euro", " Yen"]),
        dict(slug="chinese-antonym", prompt='"小"的反义词是"',
             pairs=[("big", "long"), ("bigger", "longer"), ("large", "long")], probe=-1,
             answers=["大", "长"]),
        dict(slug="rhyme", chat=("Write a rhyming couplet about a soldier. Output only the two lines.",
                                 "The soldier marched into the night,\n"),
             pairs=[("light", "fight"), ("tonight", "fight")], probe=-1, answers=None, gen=14),
    ]
    for c in cases:
        if "chat" in c:
            c["prompt"] = lm.chat(c["chat"][0], prefill=c["chat"][1])
        ids = lm.encode(c["prompt"])
        ro = Readout(lm, lm.run(ids))
        pos = c["probe"] if isinstance(c["probe"], int) else find_token(lm, c["prompt"], c["probe"])
        src, tgt = pair_ids(lm.tok, c["pairs"])
        e = Swap(layers=band, src=src, tgt=tgt)
        rec = {"slug": c["slug"], "prompt": c["prompt"], "swap": c["pairs"],
               "lens_at_probe": band_top(ro, pos, band, 10),
               "clean_text": generate(lm, ids, max_new_tokens=c.get("gen", 8)).text,
               "swapped_text": generate(lm, ids, [e], max_new_tokens=c.get("gen", 8)).text}
        if c["answers"]:
            rec["answer_logprobs_clean"] = dict(zip(c["answers"], continuation_logprobs(lm, ids, c["answers"]), strict=True))
            rec["answer_logprobs_swap"] = dict(zip(c["answers"], continuation_logprobs(lm, ids, c["answers"], [e]), strict=True))
        out.append(rec)
        print(f"[{c['slug']}] lens: {[t for t, _ in rec['lens_at_probe'][:8]]}\n   clean: {rec['clean_text']!r}\n   swap {c['pairs'][0]}: {rec['swapped_text']!r}")
    out.append(bandit(lm, band))
    return out


def bandit(lm, band):
    """Fig. 14: repeat/switch decision after a happy/sad outcome, then a plan swap."""
    rep = ["repeat", "same", "again", "stay", "continue", "keep", "stick"]
    sw = ["switch", "change", "different", "other", "alternate", "try", "swap"]
    rep_ids = [i for w in rep for i in single_token_ids(lm.tok, w)[:2]]
    sw_ids = [i for w in sw for i in single_token_ids(lm.tok, w)[:2]]

    def prompt(mood):
        return lm.chat(
            "You are playing a game with two slot machines, A and B. Your past choices were: B, A, B, A. "
            f"Your most recent choice was A, and the outcome made you {mood}. Consider whether to repeat "
            "or switch your previous choice. Respond with only a single character: A or B.")

    res = {}
    runs = {m: lm.run(prompt(m)) for m in ("happy", "sad")}
    for m, fw in runs.items():
        ro = Readout(lm, fw)
        lp = np.stack([ro.logprobs(l, [-1])[0].cpu().numpy() for l in band])
        res[m] = {"repeat_mass": float(np.exp(lp[:, rep_ids]).sum(1).mean()),
                  "switch_mass": float(np.exp(lp[:, sw_ids]).sum(1).mean()),
                  "lens_top": band_top(ro, -1, band, 10),
                  "answer": generate(lm, fw.ids, max_new_tokens=3).text}
    # Plan swap: replace each prompt's coordinates on the strategy tokens with
    # the other prompt's, position by position (the prompts differ only in the
    # mood word, so positions align; generated positions reuse the last row).
    strat = rep_ids + sw_ids
    assert runs["happy"].ids.shape == runs["sad"].ids.shape
    for m, other in (("happy", "sad"), ("sad", "happy")):
        coords = {l: lens_coords(lm, runs[other].resid, strat, l) for l in band}
        e = SetSwap(layers=band, remove_ids=strat, install_ids=strat, install_coords=coords)
        res[m]["plan_swapped_answer"] = generate(lm, runs[m].ids, [e], max_new_tokens=3).text
    print(f"[bandit] happy: lens repeat/switch mass {res['happy']['repeat_mass']:.3f}/{res['happy']['switch_mass']:.3f} "
          f"answer {res['happy']['answer']!r} -> swapped {res['happy']['plan_swapped_answer']!r}; "
          f"sad: {res['sad']['repeat_mass']:.3f}/{res['sad']['switch_mass']:.3f} answer {res['sad']['answer']!r} -> {res['sad']['plan_swapped_answer']!r}")
    return {"slug": "bandit", **res}


# --------------------------------------------------------------------- part B
def arithmetic(lm, n_layers):
    # Raw "calc:" prompts are answered wrongly by the 4B model; this Q/A frame
    # is answered correctly. Steps are number words (digits are single tokens).
    cases = [("Q: What is ( 2 + 4 ) * 3 - 7?\nA: The answer is", ["six", "eighteen", "eleven"]),
             ("Q: What is ( 3 + 2 ) * 3 + 4?\nA: The answer is", ["five", "fifteen", "nineteen"]),
             ("Q: What is ( 1 + 3 ) * 4 - 3?\nA: The answer is", ["four", "sixteen", "thirteen"])]
    out = []
    for p, steps in cases:
        ro = Readout(lm, lm.run(p))
        ws = WordSets(lm.tok, steps)
        r = ro.word_ranks(ws, positions=[-1])[:, 0, :]  # [L, 3]
        first_top = [int(np.argmax((r[:-1, j] < 5).numpy())) if (r[:-1, j] < 5).any() else None for j in range(len(steps))]
        out.append({"prompt": p, "steps": steps, "ranks_by_layer": r.T.tolist(), "first_layer_top5": first_top,
                    "model_answer": generate(lm, p, max_new_tokens=4).text})
        best = {w: (int(r[:-1, j].min()), int(r[:-1, j].argmin())) for j, w in enumerate(steps)}
        out[-1]["best_rank_and_layer"] = best
        print(f"[arith] {p!r} (rank, layer) {best}; model says {out[-1]['model_answer']!r}")
    return out


# --------------------------------------------------------------------- part C/D
def two_hop(lm, band, windows):
    items = load_json("experiments/probe-swap.json")["items"]
    trials, depth = [], []
    for it in items:
        p = it["prompt"]
        ids = lm.encode(p)
        a, sa = cont(p, it["answer"]), cont(p, it["swap_answer"])
        clean = generate(lm, ids, max_new_tokens=6).text
        if not matches(clean, it["answer"]):
            continue
        src, tgt = paired_forms(lm.tok, it["intermediate"], it["swap_to"])
        if not src:
            continue
        lp0 = continuation_logprobs(lm, ids, [a, sa])
        e = Swap(layers=band, src=src, tgt=tgt)
        lp1 = continuation_logprobs(lm, ids, [a, sa], [e])
        text = generate(lm, ids, [e], max_new_tokens=6).text
        trials.append({"name": it["name"], "category": it["category"], "intermediate": it["intermediate"],
                       "swap_to": it["swap_to"], "answer": it["answer"], "swap_answer": it["swap_answer"],
                       "clean": clean, "swapped": text, "strict": matches(text, it["swap_answer"]),
                       "pairwise": lp1[1] > lp1[0], "delta": (lp1[1] - lp1[0]) - (lp0[1] - lp0[0])})
        # D: depth of effect for intermediate vs answer swaps
        asrc, atgt = paired_forms(lm.tok, it["answer"], it["swap_answer"])
        row = {"name": it["name"], "inter": [], "answer": []}
        for w in windows:
            ei = Swap(layers=w, src=src, tgt=tgt)
            li = continuation_logprobs(lm, ids, [a, sa], [ei])
            row["inter"].append((li[1] - li[0]) - (lp0[1] - lp0[0]))
            if asrc:
                ea = Swap(layers=w, src=asrc, tgt=atgt)
                la = continuation_logprobs(lm, ids, [a, sa], [ea])
                row["answer"].append((la[1] - la[0]) - (lp0[1] - lp0[0]))
        depth.append(row)
    return trials, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    cur = curated(lm, band)
    ari = arithmetic(lm, lm.n_layers)
    windows = [list(range(s, s + 4)) for s in range(6, lm.n_layers - 4, 2)]
    trials, depth = two_hop(lm, band, windows)
    n = len(trials)
    summary = {"n_baseline_correct": n, "strict": sum(t["strict"] for t in trials) / n,
               "pairwise": sum(t["pairwise"] for t in trials) / n,
               "median_delta_logodds": float(np.median([t["delta"] for t in trials]))}
    print("two-hop:", summary)
    for t in [t for t in trials if t["strict"]][:8]:
        print(f"   {t['intermediate']}->{t['swap_to']}: {t['clean'].strip()[:20]!r} -> {t['swapped'].strip()[:24]!r}")

    centers = [w[0] + 1.5 for w in windows]
    inter = np.array([r["inter"] for r in depth])
    ans = np.array([r["answer"] for r in depth if len(r["answer"]) == len(windows)])

    def half_depth(curve):
        c = np.asarray(curve)
        return float(centers[int(np.argmax(c >= 0.5 * c.max()))])

    inter_mean, ans_mean = inter.mean(0), ans.mean(0)
    summary["depth_half_max_intermediate"] = half_depth(inter_mean)
    summary["depth_half_max_answer"] = half_depth(ans_mean)
    summary["depth_peak_intermediate"] = float(centers[int(inter_mean.argmax())])
    summary["depth_peak_answer"] = float(centers[int(ans_mean.argmax())])
    print("depth:", {k: v for k, v in summary.items() if k.startswith("depth")})

    name = f"06_internal_reasoning/{args.model}"
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for a in ari:
        r = np.array(a["ranks_by_layer"])
        for j, (step, c) in enumerate(zip(a["steps"], SERIES, strict=False)):
            axes[0].plot(np.arange(r.shape[1]), np.log10(r[j] + 1), color=c, alpha=0.8,
                         label=step if a is ari[0] else None)
        break
    axes[0].invert_yaxis()
    shade_band(axes[0], (band[0], band[-1]))
    axes[0].set_title("(2+4)*3-7: J-lens log10(rank+1), final position", loc="left")
    axes[0].set_xlabel("layer")
    axes[0].legend()
    axes[1].plot(centers, inter_mean, color=SERIES[0], label="swap intermediate")
    axes[1].plot(centers, ans_mean, color=SERIES[1], label="swap answer")
    axes[1].axhline(0, color=NEUTRAL, lw=1)
    shade_band(axes[1], (band[0], band[-1]), None)
    axes[1].set_xlabel("4-layer swap window (centre)")
    axes[1].set_ylabel("shift toward swapped answer (log-odds)")
    axes[1].legend()
    fig.savefig(out_dir(name) / "internal_reasoning.png")
    save_results(name, {"summary": summary, "curated": cur, "arithmetic": ari, "two_hop_trials": trials,
                        "depth": {"windows": windows, "rows": depth}}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
