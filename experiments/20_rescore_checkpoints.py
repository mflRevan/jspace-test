"""Final, better-powered evaluation of RL checkpoints (and the base model).

  gsm8k  -- full GSM8K test set (1319), greedy, 512 tokens
  mmlu   -- 2000 MMLU test questions, letter logits
  facts  -- single- and two-hop facts, format-robust: the gold answer (or its
            digit / number-word equivalent) appears as a word in the first
            24 generated tokens

usage: python experiments/20_rescore_checkpoints.py [CKPT_DIR ...]   (base model always included)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from jspace.evals import EvalSuite, greedy, gsm8k_reward
from jspace.io import out_dir, run_meta, seed_everything
from jspace.model import LensedModel

NUM = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def variants(ans: str) -> set[str]:
    a = ans.strip().lower()
    out = {a}
    if a.isdigit() and int(a) < len(NUM):
        out.add(NUM[int(a)])
    if a in NUM:
        out.add(str(NUM.index(a)))
    return out


def contains(text: str, ans: str) -> bool:
    words = set(re.findall(r"[\w']+", text.lower()))
    return any(v in words or (" " in v and v in text.lower()) for v in variants(ans))


def load_weights(lm: LensedModel, ckpt: Path) -> None:
    sd = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        sd.update(load_file(str(f), device="cuda"))
    # save_pretrained writes the original multimodal checkpoint layout
    sd = {k.replace("model.language_model.", "model."): v for k, v in sd.items() if not k.startswith(("model.visual", "mtp"))}
    missing, unexpected = lm.hf.load_state_dict(sd, strict=False)
    if unexpected or [m for m in missing if "lm_head" not in m]:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def main():
    torch.cuda.set_per_process_memory_fraction(0.85)
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--model", default="qwen3.5-2b")
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, dtype=torch.bfloat16, load_lens_matrices=False)
    suite = EvalSuite(lm, n_gsm8k=1319, n_mmlu=2000)
    results = {}
    for name, ckpt in [("base", None)] + [(Path(c).parent.name, Path(c)) for c in args.ckpts]:
        if ckpt is not None:
            load_weights(lm, ckpt)
        texts = greedy(lm, [p for p, _ in suite.gsm8k], 512)
        res = {"gsm8k": float(np.mean([gsm8k_reward(t, g) for t, (_, g) in zip(texts, suite.gsm8k, strict=True)])),
               "gsm8k_mean_len": float(np.mean([len(lm.tok.encode(t)) for t in texts])),
               "mmlu": suite.mmlu_acc()}
        for fname, items in suite.facts.items():
            outs = greedy(lm, [p for p, _ in items], 24)
            res[fname + "_robust"] = float(np.mean([contains(t, a) for t, (_, a) in zip(outs, items, strict=True)]))
        results[name] = res
        print(name, {k: round(v, 4) for k, v in res.items()}, flush=True)
    n = {"gsm8k": len(suite.gsm8k), "mmlu": len(suite.mmlu)}
    path = out_dir(f"20_rescore_checkpoints/{args.model}") / "results.json"
    path.write_text(json.dumps({"meta": run_meta(lm, args=vars(args)), "n": n, "results": results}, indent=1))


if __name__ == "__main__":
    main()
