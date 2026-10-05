"""Paper section 3.5.2 / Fig. 22: J-space ablation impairs internal reasoning but
leaves ordinary text prediction largely intact.

Ablation: at every token position and every layer of a range, project out the
top-10 J-lens directions (by lens logit), never ablating "protected" tokens --
the clean model's top-10 next-token predictions at that position (paper: avoid
ablating what the model intends to say). Three strengths differ in layer range
(light L17-20, medium L17-23, heavy L17-27 = whole band). Control: project out
10 random unit directions per position at the medium range.

Metrics:
* two-hop accuracy on ``probe-swap.json`` items the clean model answers
  correctly (greedy continuation starts with the answer word);
* top-1 next-token agreement with the clean model on held-out WikiText
  (32 passages x 128 tokens, positions >= 16).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from jspace.data import heldout_passages, load_json
from jspace.generate import generate
from jspace.interventions import Edit, Intervene
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, style

RANGES = {"light": (17, 20), "medium": (17, 23), "heavy": (17, 27)}


@dataclass
class TopKAblatePerPos(Edit):
    """Top-``k`` J-lens ablation with per-position protected ids ``protect[pos]``
    (``[T, m]``, absolute positions; positions beyond ``T`` use the last row)."""

    lm: LensedModel | None = None
    k: int = 10
    protect: torch.Tensor | None = None
    log: dict = field(default_factory=dict)  # (layer, pos) -> ablated ids, if pos in log_positions
    log_positions: tuple[int, ...] = ()

    def prepare(self, lm):
        self.lm = lm

    def apply(self, h, layer, pos):
        lm = self.lm
        B, P, d = h.shape
        flat = h.reshape(B * P, d)
        logits = lm.lens_logits(flat, layer)
        prot = self.protect[pos.clamp_max(self.protect.shape[0] - 1)].repeat(B, 1)  # [BP, m]
        logits.scatter_(1, prot, float("-inf"))
        ids = logits.topk(self.k, dim=-1).indices
        for i, p in enumerate(pos.tolist()):
            if p in self.log_positions:
                self.log[(layer, p)] = ids[i].tolist()
        w = lm.W_U[ids.flatten()].float() * lm.norm_gain
        V = (w @ lm.J[layer]).reshape(B * P, self.k, d)
        Q = torch.linalg.qr(V.transpose(1, 2)).Q
        proj = torch.einsum("nd,ndk->nk", flat, Q)
        return (flat - torch.einsum("nk,ndk->nd", proj, Q)).reshape(B, P, d)


@dataclass
class RandomAblate(Edit):
    """Project out ``k`` fresh random unit directions per position (control)."""

    k: int = 10
    generator: torch.Generator | None = None

    def apply(self, h, layer, pos):
        B, P, d = h.shape
        flat = h.reshape(B * P, d)
        R = torch.randn(B * P, d, self.k, device=h.device, generator=self.generator)
        Q = torch.linalg.qr(R).Q
        proj = torch.einsum("nd,ndk->nk", flat, Q)
        return (flat - torch.einsum("nk,ndk->nd", proj, Q)).reshape(B, P, d)


def make_edit(lm, cond: str, protect: torch.Tensor, gen: torch.Generator, log_positions=()):
    if cond == "clean":
        return []
    if cond == "random":
        lo, hi = RANGES["medium"]
        return [RandomAblate(layers=list(range(lo, hi + 1)), generator=gen)]
    lo, hi = RANGES[cond]
    return [TopKAblatePerPos(layers=list(range(lo, hi + 1)), protect=protect, log_positions=log_positions)]


@torch.no_grad()
def clean_protect(lm, ids: torch.Tensor, m: int = 10) -> torch.Tensor:
    logits = lm.hf(input_ids=ids[None], use_cache=False).logits[0].float()
    return logits.topk(m, dim=-1).indices  # [T, m]


def first_word(text: str) -> str:
    m = re.search(r"[\w\-']+", text)
    return m.group(0).lower() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-passages", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    lm = LensedModel(args.model, args.lens)
    gen = torch.Generator(device=lm.device).manual_seed(args.seed)
    conds = ["clean", "light", "medium", "heavy", "random"]

    # (a) two-hop reasoning
    items = load_json("experiments/probe-swap.json")["items"]
    correct = {c: [] for c in conds}
    examples = []
    kept = 0
    for it in items:
        ids = lm.encode(it["prompt"])
        prot = clean_protect(lm, ids)
        ans = it["answer"].lower()
        texts = {}
        for c in conds:
            T = ids.shape[0] - 1
            edits = make_edit(lm, c, prot, gen, log_positions=(T,))
            g = generate(lm, ids, edits, max_new_tokens=8)
            texts[c] = g.text
            if c == "clean" and not first_word(g.text).startswith(ans[:4]):
                break
            if c == "heavy" and edits:
                ablated = edits[0].log.get((20, T), [])
                texts["heavy_ablated_L20"] = [lm.tok.decode([t]).strip() for t in ablated]
        else:
            kept += 1
            for c in conds:
                correct[c].append(first_word(texts[c]).startswith(ans[:4]))
            if not correct["heavy"][-1] and len(examples) < 12:
                examples.append({"prompt": it["prompt"], "answer": it["answer"], "intermediate": it["intermediate"],
                                 **{f"text_{c}": texts[c] for c in conds}, "ablated_at_final_L20": texts.get("heavy_ablated_L20")})
    multihop = {c: float(np.mean(v)) for c, v in correct.items()}
    print(f"two-hop items answered correctly by clean model: {kept}/{len(items)}")
    print("two-hop accuracy:", {c: round(v, 3) for c, v in multihop.items()})

    # (b) ordinary text prediction
    agree = {c: [] for c in conds if c != "clean"}
    for p in heldout_passages(args.n_passages):
        ids = lm.encode(p)[:128]
        prot = clean_protect(lm, ids)
        clean_top = prot[16:-1, 0]
        for c in agree:
            with Intervene(lm, make_edit(lm, c, prot, gen)):
                lg = lm.hf(input_ids=ids[None], use_cache=False).logits[0, 16:-1]
            agree[c].append((lg.argmax(-1) == clean_top).float().mean().item())
    text_agree = {c: float(np.mean(v)) for c, v in agree.items()}
    print("wikitext top-1 agreement:", {c: round(v, 3) for c, v in text_agree.items()})

    name = f"09_jspace_ablation/{args.model}"
    style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    labels = ["light", "medium", "heavy", "random"]
    x = np.arange(len(labels))
    ax.bar(x - 0.19, [multihop[c] for c in labels], 0.36, color=SERIES[0], label="two-hop accuracy")
    ax.bar(x + 0.19, [text_agree[c] for c in labels], 0.36, color=NEUTRAL, label="WikiText top-1 agreement")
    ax.set_xticks(x, [f"{c}\n" + (f"L{RANGES[c][0]}-{RANGES[c][1]}" if c in RANGES else "rand. dirs, medium") for c in labels])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.savefig(out_dir(name) / "ablation.png")
    save_results(name, {"ranges": RANGES, "n_items_kept": kept, "multihop_accuracy": multihop,
                        "wikitext_top1_agreement": text_agree, "examples": examples}, run_meta(lm, args=vars(args)))
    for e in examples[:5]:
        print(f"\n{e['prompt']!r}\n  clean: {e['text_clean']!r}  heavy: {e['text_heavy']!r}\n  ablated@final,L20: {e['ablated_at_final_L20']}")


if __name__ == "__main__":
    main()
