"""Feasibility probe for gated RL on GSM8K (before any training).

A. Capability: greedy GSM8K-test accuracy with a short written chain of thought
   (non-thinking chat mode), answer length, truncation rate.
B. RL signal (GSM8K *train* problems): sampled pass rates (T=1, n samples) and the fraction
   of problems with mixed outcomes (the only ones giving REINFORCE signal).
C. Throughput: batched generation latency / tokens per second vs batch size.
D. Training-step cost on real sampled traces (fp32 weights, bf16 autocast,
   per-layer activation checkpointing, chunked log-probs, micro-batches of 8):
   forward+backward time and peak memory, ungated vs gated (k=10, confidence),
   plus the distribution of the gate confidence c per band layer.
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import torch
from datasets import load_dataset

from jspace.gating import WorkspaceGate
from jspace.io import run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.rl import token_logprobs, trainable_params

INSTR = "\nReason step by step, then end your reply with 'Answer: <number>'."


def gold(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def parse(text: str) -> str | None:
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        m = nums
    v = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


@torch.no_grad()
def batch_generate(lm, prompts, max_new, sample=False, n=1):
    tok = lm.tok
    tok.padding_side = "left"
    texts = [p for p in prompts for _ in range(n)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(lm.device)
    torch.cuda.synchronize()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm.hf.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=1.0 if sample else None,
                             top_p=1.0 if sample else None, top_k=0 if sample else None,
                             pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t
    new = out[:, enc.input_ids.shape[1]:]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    lens = [int(next((i for i, t in enumerate(r.tolist()) if t in eos), len(r))) for r in new]
    return [tok.decode(r, skip_special_tokens=True) for r in new], lens, dt, enc.input_ids, new


def main():
    torch.cuda.set_per_process_memory_fraction(0.85)  # fail loudly instead of spilling to host RAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--n-eval", type=int, default=1319)
    ap.add_argument("--n-sample-problems", type=int, default=128)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    seed_everything(0)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = np.random.default_rng(0).permutation(len(ds))
    lm = LensedModel(args.model, dtype=torch.bfloat16)
    chat = lambda q: lm.chat(q + INSTR)  # noqa: E731
    res = {}

    # A: greedy accuracy
    items = [ds[int(i)] for i in idx[: args.n_eval]]
    correct, lens, tokens, wall = [], [], 0, 0.0
    for s in range(0, len(items), args.batch):
        b = items[s : s + args.batch]
        texts, ls, dt, _, _ = batch_generate(lm, [chat(x["question"]) for x in b], args.max_new)
        correct += [parse(t) == gold(x["answer"]) for t, x in zip(texts, b, strict=True)]
        print(f"  A batch {s // args.batch}: acc so far {np.mean(correct):.3f} ({len(correct)}) {dt:.1f}s", flush=True)
        lens += ls
        tokens += sum(ls)
        wall += dt
    lens = np.array(lens)
    res["greedy"] = {"accuracy": float(np.mean(correct)), "n": len(items), "mean_len": float(lens.mean()),
                     "p90_len": float(np.percentile(lens, 90)), "truncated": float((lens >= args.max_new).mean()),
                     "tokens_per_s": tokens / wall, "example": texts[0][:600]}
    print("A greedy:", {k: v for k, v in res["greedy"].items() if k != "example"}, flush=True)

    # B: sampled pass rates
    train = load_dataset("openai/gsm8k", "main", split="train")
    probs = [train[int(i)] for i in np.random.default_rng(1).permutation(len(train))[: args.n_sample_problems]]
    rates, s_tokens, s_wall = [], 0, 0.0
    per_call = max(1, args.batch // args.samples)
    for s in range(0, len(probs), per_call):
        b = probs[s : s + per_call]
        texts, ls, dt, _, _ = batch_generate(lm, [chat(x["question"]) for x in b], args.max_new, sample=True, n=args.samples)
        for j, x in enumerate(b):
            rates.append(np.mean([parse(t) == gold(x["answer"]) for t in texts[j * args.samples : (j + 1) * args.samples]]))
        s_tokens += sum(ls)
        s_wall += dt
    rates = np.array(rates)
    res["sampled"] = {"pass@1": float(rates.mean()), "frac_mixed": float(((rates > 0) & (rates < 1)).mean()),
                      "frac_all_wrong": float((rates == 0).mean()), "frac_all_right": float((rates == 1).mean()),
                      "tokens_per_s": s_tokens / s_wall, "sec_per_64_rollouts": s_wall / (len(probs) * args.samples) * 64}
    print("B sampled:", res["sampled"], flush=True)

    # C: throughput vs batch size (sampled, fixed problems)
    res["throughput"] = {}
    for bs in (8, 32, 64, 128):
        b = [ds[int(i)]["question"] for i in idx[:bs]]
        _, ls, dt, _, _ = batch_generate(lm, [chat(q) for q in b], 256, sample=True)
        res["throughput"][bs] = {"sec": dt, "tokens_per_s": sum(ls) / dt}
        print(f"C batch {bs}: {dt:.1f}s for up to 256 new tokens, {sum(ls) / dt:.0f} tok/s", flush=True)

    # D: training-step cost on real traces (fp32 master weights)
    del lm
    torch.cuda.empty_cache()
    lm = LensedModel(args.model, dtype=torch.float32)
    params = trainable_params(lm)
    lm.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.hf.train()
    opt = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
    b = [ds[int(i)]["question"] for i in idx[:8]]
    _, _, _, p_ids, new = batch_generate(lm, [chat(q) for q in b], 256, sample=True, n=args.samples)
    ids = torch.cat([p_ids, new], 1)
    n_p = p_ids.shape[1]
    attn = (ids != lm.tok.pad_token_id).long()
    pos = (attn.cumsum(1) - 1).clamp_min(0)
    res["train_step"] = {"seq_len": int(ids.shape[1]), "batch": int(ids.shape[0])}
    for arm in ("baseline", "gated"):
        gate = WorkspaceGate(lm, lm.spec.band_layers, k=10, mode="jlens", confidence=True) if arm == "gated" else None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        for mb in range(0, ids.shape[0], 8):  # micro-batches of one prompt group
            x, m, ps = ids[mb : mb + 8], attn[mb : mb + 8], pos[mb : mb + 8]
            ctx = gate if gate is not None else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                hid = lm.hf.model(input_ids=x, attention_mask=m, position_ids=ps, use_cache=False).last_hidden_state
                lp = token_logprobs(lm, hid[:, n_p - 1 : -1], x[:, n_p:])
            (-(lp * m[:, n_p:]).sum() / ids.shape[0]).backward()
            del hid, lp
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        res["train_step"][arm] = {"sec": time.time() - t, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        if gate is not None:
            res["train_step"]["confidence_percentiles"] = {
                l: np.percentile(c.float().cpu().numpy(), [5, 25, 50, 75, 95]).round(3).tolist() for l, c in gate.conf.items()}
        print(f"D {arm}: {res['train_step'][arm]}", flush=True)
    print("D confidence c percentiles [5,25,50,75,95] per band layer:")
    for l, v in res["train_step"]["confidence_percentiles"].items():
        print(f"   L{l}: {v}")
    save_results(f"18_rl_feasibility/{args.model}", res, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
