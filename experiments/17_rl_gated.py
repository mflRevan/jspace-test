"""Workspace-gated vs ordinary on-policy RL: capability gain vs interference.

Train: no-CoT arithmetic "Compute a * b + c." / "a * b - c." (digits 2-9) with
REINFORCE (jspace.rl). The gated arm projects the backward gradient onto the
active J-space (top-k lens directions) at every workspace-band layer; the
baseline is identical without the gate.

Evaluated every ``--eval-every`` steps (greedy):
  id         held-out operand combinations of the training task
  transfer   a + b * c, (a + b) * c (same skill, unseen structure)
  near       a + b + c, two-digit addition (adjacent skill, retention)
  facts      two-hop factual prompts (probe-swap) and single-hop facts
             (flexible-generalization templates), accuracy
  wikitext   mean token NLL on held-out WikiText (16 x 128 tokens)
  weights    relative weight change ||W - W0|| / ||W0|| per block
The comparison of interest is the trade-off curve (gain vs interference)
across checkpoints, not a single step count.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch

from jspace.data import heldout_passages, load_json
from jspace.generate import generate
from jspace.io import out_dir, run_meta, seed_everything
from jspace.model import LensedModel
from jspace.rl import RLConfig, Trainer, reward

SUFFIX = " Answer with just the number."
COMBOS = list(itertools.product(range(2, 10), range(2, 10), range(2, 10), (0, 1)))
_r = np.random.default_rng(1234)
_perm = _r.permutation(len(COMBOS))
TRAIN_COMBOS = [COMBOS[i] for i in _perm[: int(0.7 * len(COMBOS))]]
HELD_COMBOS = [COMBOS[i] for i in _perm[int(0.7 * len(COMBOS)):]]


def arith(a, b, c, op):
    return (f"Compute {a} * {b} + {c}." if op == 0 else f"Compute {a} * {b} - {c}."), str(a * b + c if op == 0 else a * b - c)


def make_task(lm):
    def task(rng):
        q, ans = arith(*TRAIN_COMBOS[rng.integers(len(TRAIN_COMBOS))])
        return lm.chat(q + SUFFIX), ans
    return task


def eval_sets(lm, n=60, seed=7):
    rng = np.random.default_rng(seed)
    d3 = lambda: rng.integers(2, 10, size=3)  # noqa: E731
    sets = {
        "id": [arith(*HELD_COMBOS[i]) for i in rng.choice(len(HELD_COMBOS), n, replace=False)],
        "transfer_a+b*c": [(f"Compute {a} + {b} * {c}.", str(a + b * c)) for a, b, c in (d3() for _ in range(n))],
        "transfer_(a+b)*c": [(f"Compute ( {a} + {b} ) * {c}.", str((a + b) * c)) for a, b, c in (d3() for _ in range(n))],
        "near_a+b+c": [(f"Compute {a} + {b} + {c}.", str(a + b + c)) for a, b, c in (d3() for _ in range(n))],
        "near_2digit": [(f"Compute {x} + {y}.", str(x + y)) for x, y in rng.integers(10, 99, size=(n, 2))],
    }
    sets = {k: [(lm.chat(q + SUFFIX), a) for q, a in v] for k, v in sets.items()}
    sets["facts_twohop"] = [(it["prompt"].rstrip(), it["answer"]) for it in load_json("experiments/probe-swap.json")["items"]]
    single = []
    for cat in load_json("experiments/flexible-generalization.json")["categories"]:
        for fn in cat["funcs"]:
            for arg in cat["args"]:
                single.append((fn["template"].format(arg=arg), fn["answers"][arg]))
    sets["facts_single"] = single
    return sets


@torch.no_grad()
def evaluate(lm, sets, passages, W0):
    out = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name, items in sets.items():
            ok = 0
            for prompt, ans in items:
                text = generate(lm, prompt, max_new_tokens=6).text
                ok += reward(text, ans) if not name.startswith("facts") else text.strip().lower().startswith(ans.lower())
            out[name] = ok / len(items)
        nll = []
        for p in passages:
            ids = lm.encode(p)[:128]
            logits = lm.hf(input_ids=ids[None]).logits[0, :-1].float()
            nll.append(float(torch.nn.functional.cross_entropy(logits, ids[1:])))
        out["wikitext_nll"] = float(np.mean(nll))
    out["weight_change"] = []
    for blk, ws in zip(lm.layers, W0, strict=True):
        num = sum(float(((p.detach() - w0.to(p.device, p.dtype)) ** 2).sum()) for p, w0 in zip(blk.parameters(), ws, strict=True))
        den = sum(float((w0.float() ** 2).sum()) for w0 in ws)
        out["weight_change"].append((num / den) ** 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--arm", default="baseline", help="baseline or a gate mode (jlens, random, rotated, randtok)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, dtype=torch.float32)
    cfg = RLConfig(lr=args.lr, gate_mode=None if args.arm == "baseline" else args.arm, gate_k=args.k)
    trainer = Trainer(lm, cfg, seed=args.seed)
    sets = eval_sets(lm)
    passages = heldout_passages(16)
    W0 = [[p.detach().cpu().to(torch.bfloat16) for p in blk.parameters()] for blk in lm.layers]
    name = f"17_rl_gated/{args.model}/{args.arm}_lr{args.lr:g}_s{args.seed}{args.tag}"
    odir = out_dir(name)
    log = (odir / "log.jsonl").open("w")
    task = make_task(lm)
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            ev = evaluate(lm, sets, passages, W0)
            ev.update(step=step, kind="eval", wall=time.time() - t0)
            log.write(json.dumps(ev) + "\n")
            log.flush()
            print(f"[eval {step}] " + " ".join(f"{k}={v:.3f}" for k, v in ev.items() if isinstance(v, float)), flush=True)
        if step == args.steps:
            break
        st = trainer.step(task)
        st.update(step=step, kind="train", wall=time.time() - t0)
        log.write(json.dumps(st) + "\n")
        if step % 10 == 0:
            print(f"[{step}] reward={st['reward']:.3f} gnorm={st['grad_norm']:.3f} active={st['active_prompts']} t={st['wall']:.0f}s", flush=True)
    (odir / "meta.json").write_text(json.dumps(run_meta(lm, args=vars(args), config=vars(cfg)), indent=1))


if __name__ == "__main__":
    main()
