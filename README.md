# jspace

Reproducing and extending *Verbalizable Representations Form a Global Workspace
in Language Models* (Gurnee, Sofroniew, …, Lindsey; Anthropic, arXiv:2607.15495)
on open-weights models, starting with **Qwen3.5-4B** (post-trained).

The paper's Jacobian lens (J-lens) reads which tokens an internal activation is
disposed to make the model say: `lens_l(h) = W_U norm(J_l h)`, with `J_l` the
average Jacobian from layer `l` to the final layer over a text corpus. The
J-lens vectors (rows of `W_U J_l`) span the *J-space*; the paper argues it
behaves like a global workspace (reportable, steerable, carries reasoning
intermediates, broadcast, selective, capacity-limited).

Findings so far: [`reports/qwen3.5-4b_replication.md`](reports/qwen3.5-4b_replication.md).

## Layout

```
src/jspace/
  config.py          pinned model + lens revisions, measured workspace band
  model.py           LensedModel: HF model + jlens adapter + J_l on GPU, lens vectors
  readout.py         lens logits, top-k, exact full-vocab ranks over (layer, position)
  interventions.py   residual edits as forward hooks: Steer, ProjectOut, Swap
                     (pseudoinverse lens-coordinate swap), SetSwap, TopKAblate, Clamp
  generate.py        greedy decoding / scoring under interventions (KV-cache aware)
  decomposition.py   nonnegative gradient pursuit onto the J-lens dictionary
  analysis.py        band summaries, token-position lookup
  tokens.py          word -> single-token surface forms, aligned swap pairs
  data.py, io.py, plotting.py
experiments/NN_*.py  one script per paper claim; writes results/NN_*/<model>/
data/                Anthropic's released prompt sets (Apache-2.0, from
                     github.com/anthropics/jacobian-lens @ 581d398)
results/             results.json (with provenance metadata) + figures
```

## Setup

```bash
uv sync                       # Python 3.12, torch cu128, transformers>=5.5, jlens (pinned)
```

Models and lenses are pinned in `src/jspace/config.py`:

| key | model | lens |
|---|---|---|
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` @ `851bf6e` | Neuronpedia `jacobian-lens` @ `16a01f3` (jlens reference recipe, n=1000 WikiText prompts) |

Workspace band for Qwen3.5-4B: **L17-L27** of 32 (measured by `01_layer_structure.py`).

## Reproducing

Each experiment is standalone and deterministic (greedy decoding, fixed seeds):

```bash
uv run python experiments/01_layer_structure.py     # workspace band (paper 4.1)
uv run python experiments/02_lens_quality.py        # J-lens vs logit lens (A.6)
uv run python experiments/03_gallery.py             # readout gallery + interactive slices
uv run python experiments/04_verbal_report.py       # report swaps (3.1)
uv run python experiments/05_injected_thought.py    # injected-thought reports (3.1)
uv run python experiments/06_internal_reasoning.py  # intermediate swaps (3.3)
uv run python experiments/07_flexible_generalization.py  # broadcast to many functions (3.4)
uv run python experiments/08_selectivity.py [--scope all]  # flexible vs automatic (3.5.1)
uv run python experiments/09_jspace_ablation.py     # top-k J-space ablation (3.5.2)
uv run python experiments/10_capacity.py            # list capacity / eviction (4.2)
uv run python experiments/11_ignition.py            # ambiguous-input ignition (4.1.1)
uv run python experiments/12_broadcast_mlp_gain.py  # MLP gain of J-lens vectors (4.3.1)
uv run python experiments/13_occupancy.py           # occupancy / variance explained (4.2)
uv run python experiments/14_audit_and_self_monitoring.py  # alignment audit, self-monitoring (5, 6.2)
uv run python experiments/00_fit_lens.py            # fit our own lens, compare with reference
```

Every `results.json` records git commit, package versions, model/lens revisions and arguments.
Hardware used: one RTX 5090 (32 GB); the model plus lens needs ~10 GB.

## Credits

Method, prompt data and the `jlens` package are Anthropic's (Apache-2.0).
The Qwen3.5-4B lens was fitted by Neuronpedia with the reference code.
