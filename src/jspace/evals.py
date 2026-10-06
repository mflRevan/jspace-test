"""Capability and interference evals for an instruct model (greedy, batched).

  gsm8k   -- GSM8K test subset, written reasoning, 'Answer: N' (target skill)
  mmlu    -- MMLU test subset, 0-shot letter choice (broad knowledge)
  facts   -- single-hop facts (flexible-generalization templates) and two-hop
             facts (probe-swap), short continuation (factual recall)
"""

from __future__ import annotations

import re

import numpy as np
import torch
from datasets import load_dataset

from jspace.data import load_json
from jspace.model import LensedModel

GSM8K_INSTR = "\nReason step by step, then end your reply with 'Answer: <number>'."


def gsm8k_gold(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def gsm8k_parse(text: str) -> str | None:
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"-?[\d,]*\.?\d+", text)
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def gsm8k_reward(text: str, gold: str) -> float:
    return float(gsm8k_parse(text) == gold)


@torch.no_grad()
def greedy(lm: LensedModel, prompts: list[str], max_new_tokens: int, batch: int = 64) -> list[str]:
    tok = lm.tok
    tok.padding_side = "left"
    was_training = lm.hf.training
    lm.hf.eval()
    out = []
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s : s + batch], return_tensors="pt", padding=True).to(lm.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = lm.hf.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                                 use_cache=True)
        out += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    lm.hf.train(was_training)
    return out


class EvalSuite:
    def __init__(self, lm: LensedModel, *, n_gsm8k: int = 320, n_mmlu: int = 500, seed: int = 0) -> None:
        self.lm = lm
        rng = np.random.default_rng(seed)
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        self.gsm8k = [(lm.chat(gsm[int(i)]["question"] + GSM8K_INSTR), gsm8k_gold(gsm[int(i)]["answer"]))
                      for i in rng.permutation(len(gsm))[:n_gsm8k]]
        mm = load_dataset("cais/mmlu", "all", split="test")
        self.mmlu = []
        for i in rng.permutation(len(mm))[:n_mmlu]:
            q = mm[int(i)]
            body = q["question"] + "\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", q["choices"], strict=True))
            self.mmlu.append((lm.chat(body + "\nAnswer with only the letter."), "ABCD"[q["answer"]]))
        self.letter_ids = [lm.tok.convert_tokens_to_ids(L) for L in "ABCD"]
        facts = [(it["prompt"].rstrip(), it["answer"]) for it in load_json("experiments/probe-swap.json")["items"]]
        single = [(fn["template"].format(arg=a), fn["answers"][a])
                  for cat in load_json("experiments/flexible-generalization.json")["categories"]
                  for fn in cat["funcs"] for a in cat["args"]]
        self.facts = {"facts_twohop": facts, "facts_single": single}

    @torch.no_grad()
    def mmlu_acc(self) -> float:
        lm, tok = self.lm, self.lm.tok
        tok.padding_side = "left"
        was_training = lm.hf.training
        lm.hf.eval()
        correct = 0
        for s in range(0, len(self.mmlu), 32):
            b = self.mmlu[s : s + 32]
            enc = tok([p for p, _ in b], return_tensors="pt", padding=True).to(lm.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = lm.hf(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, self.letter_ids]
            correct += sum("ABCD"[int(i)] == a for i, (_, a) in zip(logits.argmax(-1), b, strict=True))
        lm.hf.train(was_training)
        return correct / len(self.mmlu)

    def run(self) -> dict:
        res = {}
        texts = greedy(self.lm, [p for p, _ in self.gsm8k], 512)
        res["gsm8k"] = float(np.mean([gsm8k_reward(t, g) for t, (_, g) in zip(texts, self.gsm8k, strict=True)]))
        res["gsm8k_mean_len"] = float(np.mean([len(self.lm.tok.encode(t)) for t in texts]))
        res["mmlu"] = self.mmlu_acc()
        for name, items in self.facts.items():
            texts = greedy(self.lm, [p for p, _ in items], 8)
            res[name] = float(np.mean([t.strip().lower().startswith(a.lower()) for t, (_, a) in zip(texts, items, strict=True)]))
        return res
