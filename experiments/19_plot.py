"""Plot / summarise runs of 19_rl_gsm8k.py.

usage: python experiments/19_plot.py RUN_DIR [RUN_DIR ...]
Writes rl_comparison.png and summary.json next to the first run directory's parent.
Panels: training reward (EMA), response length, GSM8K test accuracy vs step,
interference (MMLU, facts) vs step, and the trade-off: GSM8K gain vs mean
interference change across checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from jspace.plotting import NEUTRAL, SERIES, plt, style

INTERF = ["mmlu", "facts_twohop", "facts_single"]


def load(run: Path):
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == "train"], [r for r in rows if r["kind"] == "eval"]


def ema(x, a=0.15):
    out, m = [], x[0]
    for v in x:
        m = a * v + (1 - a) * m
        out.append(m)
    return out


def main():
    runs = [Path(p) for p in sys.argv[1:]]
    style()
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.4))
    summary = {}
    for i, run in enumerate(runs):
        tr, ev = load(run)
        c = SERIES[i]
        name = run.name
        ax[0].plot([r["step"] for r in tr], ema([r["reward"] for r in tr]), color=c, label=name)
        ax[1].plot([r["step"] for r in tr], ema([r["mean_len"] for r in tr]), color=c)
        if ev:
            steps = [e["step"] for e in ev]
            ax[2].plot(steps, [e["gsm8k"] for e in ev], marker="o", color=c)
            for j, key in enumerate(INTERF):
                ax[3].plot(steps, [e[key] - ev[0][key] for e in ev], color=c, ls=["-", "--", ":"][j], marker="o", ms=3,
                           label=f"{name}: {key}" if i == 0 or True else None)
            gain = [e["gsm8k"] - ev[0]["gsm8k"] for e in ev]
            interf = [np.mean([e[k] - ev[0][k] for k in INTERF]) for e in ev]
            ax[4].plot(interf, gain, marker="o", color=c)
            for e, x, y in zip(ev, interf, gain, strict=True):
                ax[4].annotate(str(e["step"]), (x, y), fontsize=6, color=NEUTRAL)
            summary[name] = {"evals": ev, "final_gain": gain[-1], "final_interference": interf[-1],
                             "best_gain": max(gain), "train_steps": len(tr)}
    for a, t in zip(ax, ["train reward (EMA)", "response length (EMA)", "GSM8K test accuracy",
                         "interference: change vs step 0", "trade-off: gain vs interference"], strict=True):
        a.set_title(t, loc="left")
    for a in ax[:4]:
        a.set_xlabel("step")
    ax[4].set_xlabel("mean change in MMLU / facts")
    ax[4].set_ylabel("GSM8K gain")
    ax[4].axhline(0, color=NEUTRAL, lw=1)
    ax[4].axvline(0, color=NEUTRAL, lw=1)
    ax[0].legend(fontsize=7)
    ax[3].legend(fontsize=6)
    out = runs[0].parent
    fig.savefig(out / "rl_comparison.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)})


if __name__ == "__main__":
    main()
