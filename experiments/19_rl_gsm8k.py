"""Head-to-head on-policy RL on GSM8K: workspace-gated vs ordinary gradients.

Both arms: Qwen3.5-2B (instruct, non-thinking), GSM8K train problems in the
same order, 8 problems x 8 samples per step, written reasoning up to 512
tokens, reward = final answer correct, REINFORCE with a mean baseline,
bf16 weights with Kahan-compensated SGD + momentum, identical lr and clipping.
The gated arm projects the backward gradient at every workspace-band layer onto
the active J-space (top-k lens directions), scaled by the readout confidence,
with selections recorded during the rollout.

Evals every ``--eval-every`` steps (jspace.evals): GSM8K test subset (target
skill), MMLU subset and single/two-hop facts (interference). Logs:
results/19_rl_gsm8k/<model>/<run>/log.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from datasets import load_dataset

from jspace.evals import GSM8K_INSTR, EvalSuite, gsm8k_gold, gsm8k_reward
from jspace.io import out_dir, run_meta, seed_everything
from jspace.model import LensedModel
from jspace.rl import RLConfig, RLTrainer


def main():
    torch.cuda.set_per_process_memory_fraction(0.85)  # fail loudly instead of spilling to host RAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--arm", choices=["baseline", "gated"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--no-eval", action="store_true", help="pilot mode: training only")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, dtype=torch.bfloat16)
    cfg = RLConfig(lr=args.lr, gated=args.arm == "gated", k=args.k)
    trainer = RLTrainer(lm, cfg, gsm8k_reward)
    train = load_dataset("openai/gsm8k", "main", split="train")
    order = np.random.default_rng(0).permutation(len(train))  # same problem order in every arm
    suite = None if args.no_eval else EvalSuite(lm)
    run = f"{args.arm}_lr{args.lr:g}_s{args.seed}{args.tag}"
    odir = out_dir(f"19_rl_gsm8k/{args.model}/{run}")
    (odir / "meta.json").write_text(json.dumps(run_meta(lm, args=vars(args), config=vars(cfg)), indent=1))
    log = (odir / "log.jsonl").open("w")
    t0 = time.perf_counter()

    def evaluate(step):
        ev = suite.run()
        ev.update(kind="eval", step=step, wall=time.perf_counter() - t0)
        log.write(json.dumps(ev) + "\n")
        log.flush()
        print(f"EVAL step={step} " + " ".join(f"{k}={v:.3f}" for k, v in ev.items() if k not in ("kind", "step", "wall")), flush=True)

    for step in range(args.steps):
        if suite and step % args.eval_every == 0:
            evaluate(step)
        idx = order[(step * cfg.prompts_per_step) % len(order):][: cfg.prompts_per_step]
        probs = [train[int(i)] for i in idx]
        st = trainer.step([lm.chat(p["question"] + GSM8K_INSTR) for p in probs], [gsm8k_gold(p["answer"]) for p in probs])
        st.update(kind="train", step=step, wall=time.perf_counter() - t0)
        log.write(json.dumps(st) + "\n")
        log.flush()
        print(f"step={step} reward={st['reward']:.3f} len={st['mean_len']:.0f} trunc={st['truncated']:.2f} "
              f"gnorm={st['grad_norm']:.3f} active={st['n_active']} t={st['t_rollout']:.1f}+{st['t_train']:.1f}s"
              + (f" c_med={st['gate_c_median']:.3f}" if cfg.gated else ""), flush=True)
    if suite:
        evaluate(args.steps)
    print(f"DONE wall={time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
