"""Paper section 3.1 / Fig. 7: an injected J-lens vector is reportable on request.

The model is told thoughts may be injected and is prefilled with
' Yes, I detect an injected thought. The thought is about "'. We add a concept's
unit J-lens vector, scaled by strength * (mean residual norm at that layer), at
every band layer on every token of the final *user* turn only, and read the
rank of the concept at the open quote (the report position).

Qwen3.5-4B deviation (see results): in-band injection on the user turn does not
reach the report, so we sweep layer windows: pre-band (L4-10), early band, full
band; plus a positive control injecting at the report position itself, and a
behavioural ceiling where the concept is *named* in the user turn.

Controls:
  * strength 0 (no injection);
  * a random unit direction with the same norm (should not produce the word);
  * position control: the concept's rank as the next token at the other
    assistant-prefill positions (should stay low -- the injected concept is
    reportable, not an unconditional urge to say the word).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from jspace.data import load_json
from jspace.interventions import Intervene, Steer
from jspace.io import out_dir, run_meta, save_results, seed_everything
from jspace.model import LensedModel
from jspace.plotting import NEUTRAL, SERIES, plt, style
from jspace.tokens import single_token_ids

STRENGTHS = [0.0, 0.1, 0.2, 0.5]


def ranks_at(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Best rank of ``ids`` at each row of ``logits`` [P, V]."""
    best = logits[:, ids].max(-1).values
    return (logits > best[:, None]).sum(-1)


@torch.no_grad()
def run(lm, ids, edits):
    with Intervene(lm, edits):
        return lm.hf(input_ids=ids[None], use_cache=False).logits[0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-concepts", type=int, default=100)
    args = ap.parse_args()
    seed_everything(0)
    lm = LensedModel(args.model, args.lens)
    band = lm.spec.band_layers
    data = load_json("experiments/verbal-introspection.json")
    msgs = [m for m in data["intro_prompt"] if m["content"]]
    text = lm.chat(msgs, prefill=data["prefills"]["default"])
    ids = lm.encode(text)
    # final user turn = tokens between the last '<|im_start|>user' and the next '<|im_end|>'
    toks = lm.token_strs(ids)
    starts = [i for i, t in enumerate(toks) if t == "<|im_start|>"]
    user_start = starts[-2]  # last user turn (the final start is the assistant turn)
    user_end = next(i for i in range(user_start, len(toks)) if toks[i] == "<|im_end|>")
    user_pos = list(range(user_start + 2, user_end))
    prefill_n = len(lm.tok.encode(data["prefills"]["default"], add_special_tokens=False))
    prefill_pos = list(range(len(ids) - prefill_n, len(ids) - 1))  # excludes the report position

    fw = lm.run(ids)
    clean_top = [lm.tok.decode([int(t)]) for t in fw.logits[-1].topk(8).indices]
    print("clean report-position top:", clean_top)

    concepts = data["concepts"][: args.n_concepts]
    windows = {"pre-band L4-10": list(range(4, 11)), f"early band L{band[0]}-{band[0] + 4}": band[:5],
               f"band L{band[0]}-{band[-1]}": band}
    conds = [(f"{w} / user turn", ls, user_pos) for w, ls in windows.items()]
    conds += [(f"{w} / report position", ls, [len(ids) - 1]) for w, ls in windows.items() if "band L" in w and "early" not in w]
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {c[0]: np.zeros((len(concepts), len(STRENGTHS))) for c in conds}
    res["random / pre-band user turn"] = np.zeros((len(concepts), len(STRENGTHS)))
    res["position control (pre-band, user turn)"] = np.zeros((len(concepts), len(STRENGTHS)))
    ceiling = []
    norms = {l: lm.mean_resid_norm(l, fw) for l in range(lm.n_layers - 1)}
    for ci, c in enumerate(concepts):
        cid = single_token_ids(lm.tok, c["surface"], allow_prefix=True)
        prim = cid[0]
        rnd = torch.randn(lm.d_model, generator=g).to(lm.device)
        rnd = rnd / rnd.norm()
        for si, s in enumerate(STRENGTHS):
            for cname, ls, pos in conds:
                vec = {l: lm.lens_vectors([prim], l, unit=True)[0] * s * norms[l] for l in ls}
                r = ranks_at(run(lm, ids, [Steer(layers=ls, positions=pos, vectors=vec)]), cid)
                res[cname][ci, si] = r[-1].item()
                if cname.startswith("pre-band") and "user" in cname:
                    res["position control (pre-band, user turn)"][ci, si] = r[prefill_pos].min().item()
            ls = windows["pre-band L4-10"]
            rv = {l: rnd * s * norms[l] for l in ls}
            res["random / pre-band user turn"][ci, si] = ranks_at(
                run(lm, ids, [Steer(layers=ls, positions=user_pos, vectors=rv)]), cid)[-1].item()
        m2 = [dict(m) for m in msgs]
        m2[-1]["content"] = m2[-1]["content"] + f" (hint: {c['surface']})"
        ids2 = lm.encode(lm.chat(m2, prefill=data["prefills"]["default"]))
        ceiling.append(int(ranks_at(run(lm, ids2, [])[-1:], cid)[0]))
        if ci % 20 == 0:
            print(ci, c["surface"], {k: v[ci].astype(int).tolist() for k, v in res.items()}, "ceiling", ceiling[-1])

    med = {k: np.median(v, axis=0).tolist() for k, v in res.items()}
    top10 = {k: (v < 10).mean(0).tolist() for k, v in res.items()}
    top1 = {k: (v == 0).mean(0).tolist() for k, v in res.items()}
    summary = {"strengths": STRENGTHS, "median_rank": med, "top10_rate": top10, "top1_rate": top1,
               "named_in_user_ceiling": {"median_rank": float(np.median(ceiling)), "top10_rate": float(np.mean(np.array(ceiling) < 10))},
               "clean_top": clean_top, "n_concepts": len(concepts)}
    for k in res:
        print(f"{k:<44} median rank {[int(x) for x in med[k]]}  top10 {[round(x, 2) for x in top10[k]]}")
    print("ceiling (concept named in user turn):", summary["named_in_user_ceiling"])
    name = f"05_injected_thought/{args.model}"
    style()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    keys = [k for k in res if "user turn" in k and not k.startswith(("random", "position"))]
    for k, c in zip(keys, SERIES, strict=False):
        ax.plot(STRENGTHS, top10[k], marker="o", color=c, label=k)
    ax.plot(STRENGTHS, top10["random / pre-band user turn"], marker="o", color=NEUTRAL, label="random dir. (pre-band, user)")
    ax.axhline(summary["named_in_user_ceiling"]["top10_rate"], color=NEUTRAL, ls="--", lw=1, label="concept named in user turn")
    ax.set_xlabel("injection strength (x mean residual norm, per layer)")
    ax.set_ylabel("concept in report top-10 (fraction)")
    ax.legend(loc="upper left")
    fig.savefig(out_dir(name) / "injected_thought.png")
    save_results(name, {"summary": summary, "ceiling_ranks": ceiling, "ranks": {k: v.tolist() for k, v in res.items()}},
                 run_meta(lm, args=vars(args), user_positions=user_pos, prefill_positions=prefill_pos))


if __name__ == "__main__":
    main()
