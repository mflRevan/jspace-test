"""Fit a Jacobian lens with the reference recipe (jlens.fit) and compare it with
the reference lens for the same model.

Recipe (paper A.7 / jlens defaults): WikiText-103 *train* records >= 600 chars,
128 tokens, cotangent injected at every position >= 16 of the final block's
output, averaged over source positions, then over prompts.

Comparison against the default lens of the model: per-layer cosine between the
J matrices and top-10 readout agreement on held-out text.
"""

from __future__ import annotations

import argparse

import jlens
import numpy as np
import torch
from jlens.examples import load_wikitext_prompts

from jspace.config import LENS_DIR
from jspace.data import heldout_passages
from jspace.io import run_meta, save_results, seed_everything
from jspace.model import LensedModel, load_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--offset", type=int, default=1000, help="skip the first N records (used by the reference fit)")
    ap.add_argument("--dim-batch", type=int, default=8)
    args = ap.parse_args()
    seed_everything(0)
    jlens.configure_logging()
    lm = LensedModel(args.model, load_lens_matrices=False)
    prompts = load_wikitext_prompts(args.offset + args.n_prompts)[args.offset:]
    out = LENS_DIR / lm.spec.lenses["local"].local
    out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(lm.lm, prompts, dim_batch=args.dim_batch, max_seq_len=128,
                     checkpoint_path=str(out) + ".ckpt", checkpoint_every=4)
    lens.save(str(out))

    payload = {"n_prompts": lens.n_prompts, "path": str(out)}
    if lm.spec.default_lens != "local":
        ref = load_lens(lm.spec)
        cos = {l: float(torch.nn.functional.cosine_similarity(lens.jacobians[l].flatten(), ref.jacobians[l].float().flatten(), dim=0))
               for l in lens.source_layers}
        lm.J = {l: J.to(lm.device) for l, J in lens.jacobians.items()}
        ref_J = {l: J.float().to(lm.device) for l, J in ref.jacobians.items()}
        agree = {l: [] for l in lm.spec.band_layers}
        for text in heldout_passages(16):
            fw = lm.run(lm.encode(text)[:128])
            for l in agree:
                a = lm.lm.unembed(fw.resid[l, 16:].float() @ lm.J[l].T).topk(10).indices
                b = lm.lm.unembed(fw.resid[l, 16:].float() @ ref_J[l].T).topk(10).indices
                agree[l].append(np.mean([len(set(x.tolist()) & set(y.tolist())) / 10 for x, y in zip(a, b, strict=True)]))
        payload.update(cosine_vs_reference=cos, top10_overlap_band={l: float(np.mean(v)) for l, v in agree.items()})
        print("cosine (band):", {l: round(cos[l], 3) for l in lm.spec.band_layers})
        print("top-10 overlap (band):", {l: round(v, 3) for l, v in payload["top10_overlap_band"].items()})
    save_results(f"00_fit_lens/{args.model}", payload, run_meta(lm, args=vars(args)))


if __name__ == "__main__":
    main()
