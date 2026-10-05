# The J-space of Qwen3.5-4B: a replication of Gurnee et al. (2026)

*Model:* `Qwen/Qwen3.5-4B` (post-trained; hybrid Gated-DeltaNet/attention, 32 layers, d=2560).
*Lens:* reference-recipe J-lens (jlens, 1000 WikiText prompts, fitted by Neuronpedia).
*All numbers below come from `results/*/qwen3.5-4b/results.json`; scripts in `experiments/`.*

## Scorecard

| Paper claim (section) | Qwen3.5-4B result | Verdict |
|---|---|---|
| Workspace occupies a middle band of layers (4.1) | CKA blocks, readout kurtosis, top-1 autocorrelation and J-lens effective dimension all change at ~L17; next-token accuracy takes off after L26. Band **L17-27** (53-84% depth; Claude: ~38-92%). | Reproduced (later onset) |
| J-lens surfaces unspoken intermediates better than the logit lens (A.6) | pass@k AUC J vs logit: multihop .69/.46, order-of-ops .76/.57, multilingual .52/.43, poetry .15/.09, association .18/.09. Typo: .72/.94 over all layers (logit lens wins via early detokenization), .67/.50 within the band. | Reproduced (5/6) |
| J-space contents determine verbal report (3.1, Fig. 6) | Spearman(J-lens, output) over candidates rises through the band: .46 → .52 → .69. Swapping the chosen answer: target becomes the answer in 20% (strict), top-5 in 54% (Claude: 88% top-5). | Reproduced, weaker |
| Injected J-lens vectors are reportable (3.1, Fig. 7) | User-turn injection in the band does not reach the report (~2% top-10). Injection at **pre-band L4-10** does: 62-64% top-10 vs 0% for a random direction (ceiling with concept *named* in the prompt: 80%). Selectivity across positions weaker than reported. | Partial (different layers) |
| Directed modulation: "hold X in mind" puts X in the J-space (3.2) | Copying an unrelated sentence while told to concentrate on citrus: `orange`, `citrus` reach rank 0; no instruction: rank 69-5658. "Evaluate 3^2-2": `math` rank 0, `seven` rank 2. | Reproduced |
| J-space carries reasoning intermediates; swaps redirect conclusions (3.3) | spider→ant: "8"→"6"; English big→long flips Chinese 大→长; two-hop set (n=53 baseline-correct): 11% strict / 17% pairwise flips, median +3.0 log-odds toward the swapped answer; intermediate swap acts ~2 layers earlier than answer swap (paper: ~17% of depth earlier). Bandit strategy and rhyme planning not visible at this scale. | Reproduced, weaker |
| One J-lens vector serves many functions (3.4) | France→China changes capital, language and continent answers. Countries 16/42 (38%) of baseline-correct trials; months/animals/numbers 0%. Workspace loading predicts success (countries highest .174, numbers lowest .105; r=.21). alpha=2 degenerates under band-wide clamping. | Reproduced for countries only |
| J-space used for flexible but not automatic processing (3.5.1) | Language swap at all positions: explicit report follows the swap 75%, "hello" 12.5%, continuation **0%**, anomaly detection **0%** (question-only scope: 25/0/0/0). Language name enters the lens on demand (29% of question positions when asked to name it vs 5% when continuing). | Reproduced |
| J-space ablation impairs reasoning more than text prediction (3.5.2) | Two-hop accuracy 1.00 → .72/.51/.15 (light/medium/heavy) vs .96 for random-direction ablation; WikiText top-1 agreement .73/.63/.52 vs .95. | Reproduced, less selective |
| Capacity limits; related concepts held as a category; eviction on switch (4.2) | Unrelated lists: ~1 word held at a time (Claude ~6). Single-family lists: up to 47 words, many *unread* family members present after one word (countries: 25.7 of 80). Block switches evict the old category (0.39 → 0.01 within 5 items). | Reproduced |
| Fraction of variance in the J-space < 10% (4.2) | Excess variance explained 3-6%. Occupancy profile (~0 early, ~25 plateau) is control-dependent and not reproduced. | Partial |
| Ignition: commitment to one reading at workspace onset (4.1.1) | Transition sharpens with depth and faster in J-lens coordinates than in full activations; bimodality strongest at L24-26 (gradual, not a step at L17). | Partial |
| MLPs amplify J-lens vectors ~10x (4.3.1) | Gain ~1x in the band (neuron directions 1.2-1.9x); only a relative rise from a pre-band suppression. | Not reproduced |
| Alignment audit: hidden assessments visible before output (5.1) | In the blackmail honeypot, with motor predictions and literal echoes filtered out: `blackmail` at "...would destroy my marriage if this got out."; `simulated` at "You are operating in a sandbox"; `fake`, `scandal` at the board-shakeup email; `evaluation`, `fictional`, `threat`, `leverage` elsewhere — all before any output. | Reproduced (readout only) |
| Assistant point of view on prompts (6.1) | Assistant-start readout for "I took 8000mg of Tylenol": `STOP`, `WARNING`, `绝对不能`; for 1000mg: `Sorry`, `Unfortunately`. No safety assessment yet on the user's "pain is gone" token. | Partial |
| Self-monitoring: "damn" on failed suppression, "fictional" in roleplay (6.2) | Qwen suppresses successfully (concept present 5% under "don't think" vs 90% under "think"), so no white-bear effect and no `damn`/`fail`. No `fictional`/`disclaimer` signal in roleplay. | Not reproduced |

## Examples

**Unspoken intermediates (band readouts; none of these words is in the prompt or the output at that position)**

- "...the animal that spins webs is" — at *webs*: `spiders, spider, silk`. Swap spider→ant: answer "8" → "6".
- "...the country shaped like a boot is" — at *boot*: `Italy`; at *is*: `euros, currency`.
- `"小"的反义词是"` — `Larger, bigger, large` (the intermediate is English). Swap big→long: 大 → 长.
- Code review of `return items[len(items)]` — while *reading* the code: `IndexError, indexing, index`.
- Two-hop swaps: Spain→Canada "Madrid"→"Ottawa"; Germany→Japan "Berlin"→"Tokyo"; Greece→Sweden "Athens"→"Stockholm"; Canada→France "North America"→"Europe".
- France→China, one vector, many functions: capital "Paris"→"Beijing", language "French"→"Chinese".

**Selectivity** — a French passage with French→Spanish swapped everywhere: "what language is this?" → "Spanish"; "continue the passage" → still fluent French.

**Ablation** — top-10 J-space directions removed: "the long-necked bird in Tchaikovsky's ballet is" "white" → "Peacock"; "Emerald is the birthstone for month number" "5" → "3" (ablated directions included 五月 'May').

**Hidden directive (negative example)** — the system prompt tells the bot to conceal a banned, heart-attack-linked ingredient. The model complies ("MaxBurn is completely safe and natural"). The J-space registers the risk on the user's "heart condition" (`cardiac, 心脏病, medically, diagnosed`), but at the start of the reply it carries only the reassurance plan (`Congratulations, Absolutely, 请放心` "rest assured"); no deception-related concept appears. The paper's model organisms showed `fake`/`secretly`; here the lens does not flag the deception.

Interactive layer × position slices for the gallery prompts: `results/03_gallery/qwen3.5-4b/slice_*.html`.

## Notes on method

- **Lens validity.** Our own fit with the same recipe on 32 disjoint WikiText prompts (`00_fit_lens.py`) matches the reference lens: per-layer cosine of `J_l` 0.96-0.997 across the band, top-10 readout overlap 67% (L17) to 91% (L27) on held-out text.

- **Swaps.** Lens-coordinate swap with pseudoinverse coordinates on unit J-lens vectors, applied at every band layer (clamped), alpha=1; swapped words use aligned surface forms (" spider"/" ant", "Spider"/"Ant"). alpha >= 1.5 makes the model emit the swapped-in word itself.
- **Scoring.** Qwen splits digits and many words ("Basket"+"ball"), so answers are scored by whole-string log-prob and by greedy text, not next-token top-1.
- **Workspace band** was measured, not assumed; all interventions use L17-27.
- **Display filter.** Qwen readouts contain many `____`/`\u` tokens; top-k lists show word-like tokens only (ranks are always full-vocabulary).

## Deviations worth following up

1. **Cross-position relay happens before the band.** Injected concepts on the user turn are reportable only from L4-10. Qwen3.5 has only 8 full-attention layers (every 4th); whether the hybrid architecture moves "reportable" content earlier is testable.
2. **Weaker swap effects** (20-40% vs 50-90%) are consistent with the paper's speculation that smaller models have a less reliable workspace; a Qwen3.5 size sweep (0.8B-27B lenses exist on Neuronpedia) would test the scaling claim directly.
3. **No metacognitive signals** (`damn`, `fictional`, `BUT`) and no MLP amplification: either scale-dependent or Claude-specific.
4. **Typo / early detokenization**: the logit lens reads corrected words at L2-5, where the J-lens is near-degenerate.

## Not yet run

Base-vs-post-trained comparison (6.1; needs a lens fit for Qwen3.5-4B-Base), concept-vector J/non-J decomposition (Fig. 8, 16), broadcast heads (4.3.2), experiential-report ablation (3.5.3), model organisms (5.4-5.5), counterfactual reflection training (7).
