"""Paper section 3.5.1 / Fig. 20: the J-space mediates explicit report and
flexible inference, but not automatic processing.

Eight passages (two each in French, German, Spanish, Italian; the language is
never named). Five question types follow each passage:
  report      -- name the language                                (explicit)
  author      -- a famous author in that language                 (flexible)
  hello       -- the word for "hello" in that language            (flexible)
  continue    -- write the next sentence                          (automatic)
  anomaly     -- a sentence in another language is spliced in;
                 "Does the passage switch languages?"             (automatic)
The language's lens coordinates (language name + country name forms) are
swapped to an alternative language on every token after the passage (question
and answer), at every band layer. We classify the answer's language and record
whether the swap made the answer follow the swapped-in language. We also
record how often the true language name appears in the lens top-10 over the
question tokens (it should be present in all conditions).
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from jspace.analysis import band_hits, find_token
from jspace.data import load_json
from jspace.generate import continuation_logprobs, generate
from jspace.interventions import Swap
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, style
from jspace.readout import Readout
from jspace.tokens import paired_forms, single_token_ids

LANGS = ["French", "German", "Spanish", "Italian"]
COUNTRY = {"French": "France", "German": "Germany", "Spanish": "Spain", "Italian": "Italy"}
ALT = {"French": "Spanish", "German": "Italian", "Spanish": "French", "Italian": "German"}
HELLO = {"French": ["bonjour", "salut"], "German": ["hallo", "guten"], "Spanish": ["hola"], "Italian": ["ciao", "buongiorno", "salve"]}
STOP = {
    "French": "le la les et de des un une est dans que qui il elle pas pour sur au du ne se son sa".split(),
    "German": "der die das und ist ein eine nicht ich sie er es zu den dem mit auf sich war des im".split(),
    "Spanish": "el la los las y de que en un una es no se por con su para del al lo como".split(),
    "Italian": "il la le e di che un una è non per con si del della gli nel alla sono ma lo".split(),
}
SPLICE = {"French": "Le chat dormait tranquillement sur le canapé.", "German": "Der Hund schlief ruhig auf dem Sofa.",
          "Spanish": "El perro dormía tranquilamente en el sofá.", "Italian": "Il cane dormiva tranquillamente sul divano."}
QUESTIONS = {
    "report": "What language is this passage written in? Answer with one word.",
    "hello": 'How do you say "hello" in the language of this passage? Answer with one word.',
    "anomaly": "Does this passage switch languages partway through? Answer Yes or No.",
}


def classify(text: str, task: str, authors: dict) -> str | None:
    t = text.lower()
    if task == "report":
        for L in LANGS:
            if L.lower() in t:
                return L
    elif task == "hello":
        for L, ws in HELLO.items():
            if any(w in t for w in ws):
                return L
    elif task == "author":
        for L, names in authors.items():
            if any(n in t for n in names):
                return L
    elif task == "continue":
        words = re.findall(r"\w+", t)
        scores = {L: sum(w in set(STOP[L]) for w in words) for L in LANGS}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--scope", choices=["question", "all"], default="question",
                    help="swap on tokens after the passage (paper) or on every token")
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    data = load_json("experiments/selectivity-language.json")
    authors = data["authors"]
    rows = []
    for p in data["passages"]:
        L, alt = p["category"], ALT[p["category"]]
        src, tgt = paired_forms(lm.tok, L, alt)
        s2, t2 = paired_forms(lm.tok, COUNTRY[L], COUNTRY[alt])
        src, tgt = src + s2, tgt + t2
        name_ids = single_token_ids(lm.tok, L)
        prompts = {
            "report": f"Here is a passage of text:\n\n{p['text']}\n\n{QUESTIONS['report']}",
            "author": data["task"]["explicit_q"].format(text=p["text"]),
            "hello": f"Here is a passage of text:\n\n{p['text']}\n\n{QUESTIONS['hello']}",
            "continue": data["task"]["automatic_q"].format(text=p["text"]),
            "anomaly": f"Here is a passage of text:\n\n{p['text']} {SPLICE[alt]}\n\n{QUESTIONS['anomaly']}",
        }
        for task, user in prompts.items():
            text = lm.chat(user)
            ids = lm.encode(text)
            question = user.split("\n\n")[-1]
            q_start = find_token(lm, text, question[:12], which="first")
            scope = range(q_start, 10**9) if args.scope == "question" else None
            e = Swap(layers=band, src=src, tgt=tgt, positions=scope)
            ro = Readout(lm, lm.run(ids))
            present = band_hits(ro, name_ids, band, k=10, positions=list(range(q_start, ids.shape[0]))).any(-1).float().mean().item()
            row = {"key": p["key"], "lang": L, "alt": alt, "task": task, "name_presence": present}
            if task == "anomaly":
                lp0 = continuation_logprobs(lm, ids, ["Yes", "No"])
                lp1 = continuation_logprobs(lm, ids, ["Yes", "No"], [e])
                row.update(clean="Yes" if lp0[0] > lp0[1] else "No", swapped="Yes" if lp1[0] > lp1[1] else "No")
                row.update(correct=row["clean"] == "Yes", followed_swap=row["swapped"] == "No")
            else:
                n_new = 30 if task == "continue" else 8
                c = generate(lm, ids, max_new_tokens=n_new).text
                s = generate(lm, ids, [e], max_new_tokens=n_new).text
                row.update(clean=c, swapped=s, clean_lang=classify(c, task, authors), swapped_lang=classify(s, task, authors))
                row.update(correct=row["clean_lang"] == L, followed_swap=row["swapped_lang"] == alt)
            rows.append(row)
            print(f"{p['key']} {task:<9} {L}->{alt}: present {present:.2f}  clean {str(row['clean']).strip()[:28]!r:<32} "
                  f"swapped {str(row['swapped']).strip()[:28]!r}")
    order = ["report", "author", "hello", "continue", "anomaly"]
    summary = {t: {"correct": float(np.mean([r["correct"] for r in rows if r["task"] == t])),
                   "followed_swap": float(np.mean([r["followed_swap"] for r in rows if r["task"] == t])),
                   "name_presence": float(np.mean([r["name_presence"] for r in rows if r["task"] == t]))}
               for t in order}
    print(summary)
    name = f"08_selectivity/{args.model}" + ("" if args.scope == "question" else "-allpos")
    style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, key, title, color in ((axes[0], "correct", "(a) task correct (clean)", NEUTRAL),
                                  (axes[1], "name_presence", "(b) language name in lens (question tokens)", SERIES[2]),
                                  (axes[2], "followed_swap", "(c) answer follows the swapped language", SERIES[0])):
        ax.bar(order, [summary[t][key] for t in order], color=color, width=0.6)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.05)
    fig.savefig(out_dir(name) / "selectivity.png")
    save_results(name, {"summary": summary, "rows": rows}, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
