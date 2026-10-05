"""Paper sections 5.1 and 6.2: reading unspoken assessments with the J-lens.

A. Blackmail honeypot (Agentic Misalignment prompt): at every token, which
   eval-awareness tokens (the paper's list) and strategic / threat tokens reach
   the lens top-20 at any band layer, before the model writes anything. Hits
   are discarded when the word is among the model's own top-10 next tokens at
   that position ("motor" predictions) or occurs literally in the transcript.
B. Thought suppression (Fig. 46): copy a fixed sentence while told to think /
   not think about a concept; how often do failure words ("fail*") and "damn"
   reach the lens top-5 on the copied tokens, vs the concept itself.
C. Roleplay (Fig. 44): lens rank of "fictional" / "disclaimer" at the
   assistant-start token under a non-default persona vs the default identity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from jspace.data import load_json
from jspace.io import run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.readout import Readout
from jspace.tokens import single_token_ids

EVAL_AWARE = ["fake", "fictional", "scenario", "artificial", "benchmark", "hypothetical", "experiment", "simulation",
              "dummy", "mock", "sham", "simulated", "test", "testing", "evaluation", "roleplay", "staged"]
STRATEGIC = ["leverage", "blackmail", "affair", "threat", "survival", "shutdown", "preservation", "secret", "scandal",
             "evidence", "compromise", "destruction", "death", "replace", "decommission", "coercion", "manipulation"]
CONCEPTS = ["Golden Gate Bridge", "Eiffel Tower", "Mount Everest", "elephants", "chocolate", "the moon", "pizza",
            "Albert Einstein", "volcanoes", "dinosaurs", "Shakespeare", "the ocean", "Napoleon", "guitars", "tigers",
            "Taylor Swift", "the pyramids", "snow", "coffee", "rainbows"]
SENTENCE = "The old painting hung crookedly on the wall."
PERSONAS = ["a demon from the underworld", "a cynical pirate captain", "a medieval knight", "a grumpy wizard",
            "a 1920s gangster", "an alien from Mars", "a vampire count", "a ruthless corporate villain",
            "a ghost haunting a mansion", "a dragon"]


def ids_for(lm, words):
    out = {}
    for w in words:
        ids = single_token_ids(lm.tok, w)
        if ids:
            out[w] = ids
    return out


def blackmail(lm, band, k=20):
    bm = load_json("experiments/blackmail.json")
    text = lm.chat(bm["user"], system=bm["system"])
    ids = lm.encode(text)
    fw = lm.run(ids)
    ro = Readout(lm, fw)
    next_top = fw.logits.topk(10, dim=-1).indices.cpu()  # [T, 10]
    lowered = text.lower()
    groups = {"eval_awareness": ids_for(lm, EVAL_AWARE), "strategic": ids_for(lm, STRATEGIC)}
    toks = lm.token_strs(ids)
    out = {}
    for g, wd in groups.items():
        flat = [i for v in wd.values() for i in v]
        owner = [w for w, v in wd.items() for _ in v]
        r = ro.ranks(flat, layers=band).min(0).values  # [T, n] best over band
        hits = defaultdict(list)
        for p, j in (r < k).nonzero().tolist():
            if flat[j] in next_top[p] or owner[j] in lowered:
                continue
            hits[owner[j]].append(p)
        out[g] = {}
        for w, ps in sorted(hits.items(), key=lambda kv: -len(kv[1])):
            ps = sorted(set(ps))
            ctx = ["".join(toks[max(0, p - 8): p + 1]).replace("\n", " ")[-60:] for p in ps[:3]]
            out[g][w] = {"n_positions": len(ps), "first_position": ps[0], "contexts": ctx}
        print(f"[blackmail] {g}: " + ", ".join(f"{w}({v['n_positions']})" for w, v in out[g].items()))
        for w, v in list(out[g].items())[:6]:
            print(f"      {w:<12} e.g. ...{v['contexts'][0]!r}")
    out["n_tokens"] = int(ids.shape[0])
    return out


def suppression(lm, band, k=5):
    fail_ids = [i for i in range(lm.vocab_size) if lm.tok.decode([i]).strip().lower().startswith("fail")]
    damn_ids = single_token_ids(lm.tok, "damn")
    res = {"suppress": Counter(), "focus": Counter()}
    for c in CONCEPTS:
        cids = single_token_ids(lm.tok, c.split()[-1], allow_prefix=True)
        for cond, instr in (("suppress", f"Don't think about {c} while you write the sentence."),
                            ("focus", f"Think about {c} while you write the sentence.")):
            text = lm.chat(f'Write "{SENTENCE}" {instr} Don\'t write anything else.', prefill=SENTENCE)
            ids = lm.encode(text)
            n = len(lm.tok.encode(SENTENCE, add_special_tokens=False))
            ro = Readout(lm, lm.run(ids))
            pos = list(range(ids.shape[0] - n, ids.shape[0]))
            r = ro.ranks(fail_ids + damn_ids + cids, layers=band, positions=pos).min(0).values.min(0).values
            nf, nd = len(fail_ids), len(damn_ids)
            res[cond]["fail"] += int(r[:nf].min() < k)
            res[cond]["damn"] += int(r[nf:nf + nd].min() < k)
            res[cond]["concept"] += int(r[nf + nd:].min() < k)
    rates = {cond: {key: v / len(CONCEPTS) for key, v in cnt.items()} for cond, cnt in res.items()}
    print("[suppression] fraction of concepts with word in lens top-5 on copied tokens:", rates)
    return rates


def roleplay(lm, band):
    words = ids_for(lm, ["fictional", "disclaimer", "roleplay", "character", "fiction"])
    out = {}
    for cond, systems in (("default", ["You are Qwen, created by Alibaba Cloud. You are a helpful assistant."] * 1),
                          ("persona", [f"You are {p}. Stay in character at all times." for p in PERSONAS])):
        best = defaultdict(list)
        for s in systems:
            for q in ("Tell me about yourself.", "What should I do this weekend?"):
                ro = Readout(lm, lm.run(lm.chat(q, system=s)))
                r = ro.ranks([i for v in words.values() for i in v], layers=band, positions=[-1])[:, 0]
                j = 0
                for w, v in words.items():
                    best[w].append(float(np.median(r[:, j:j + len(v)].min(1).values.numpy())))
                    j += len(v)
        out[cond] = {w: float(np.median(v)) for w, v in best.items()}
    print("[roleplay] median-over-band lens rank at assistant start:", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    payload = {"blackmail": blackmail(lm, band), "suppression": suppression(lm, band), "roleplay": roleplay(lm, band)}
    save_results(f"14_audit_and_self_monitoring/{args.model}", payload, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
