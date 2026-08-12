# Pretrain + SFT scale analysis: capacity vs. undertraining

Date: 2026-08-12

## Question

After the scale-up in `docs/superpowers/specs/2026-08-09-model-scale-up-design.md`
(75.5M → 220.2M non-embedding params), both `generate.py` runs against the new
checkpoints — the raw pretrained model (`my_checkpoints/step_10500.pt`) and the SFT'd
model (`my_sft_checkpoints/step_12000.pt`) — still produce incoherent or factually wrong
output ("the capital of France is Paris City (Canton)..."). This doc answers: **did we not
train long enough, or does the model still lack capacity?** — using the actual W&B run
data (`objones25/llm-training/5n1puqxx` pretrain, `objones25/llm-training/mon0fm4y` SFT)
plus a direct weight/activation inspection of both checkpoints.

**Conclusion up front: capacity, not training duration.** Details and evidence below.
Plots referenced throughout are saved to `plots/`.

## 1. The runs

|                           | Pretrain (`5n1puqxx`, `summer-sun-23`)                                                                  | SFT (`mon0fm4y`, `revived-gorge-24`)                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Dataset                   | `fineweb_edu`                                                                                           | `smoltalk`, init from the pretrain checkpoint                        |
| Steps                     | 10,500                                                                                                  | 12,000                                                               |
| Tokens                    | 32×8×2048×10,500 = **5.505B**                                                                           | — (loss only over unmasked assistant turns, not directly comparable) |
| Final `loss` / `val_loss` | 3.027 / 3.005                                                                                           | 1.079 / 1.118                                                        |
| Model                     | 253.76M total / **220.2M non-embedding** params (`d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4`) | same, weights loaded via `--init-from-checkpoint`                    |

`plots/pretraining-scaleup.png` and `plots/sft-scaleup.png` show `loss`/`val_loss`/`lr`/`grad_norm`
for the full run (for a direct before/after comparison against the prior architecture's
curves, saved earlier as `plots/pretraining.png` / `plots/sft.png`). Both new curves are
smooth, monotonic, and show no train/val divergence — nothing here points to an
optimization problem (bad LR, instability, overfitting).

## 2. Chinchilla check

Tokens per non-embedding parameter: 5,505,024,000 / 220,201,984 ≈ **25.0**.
Chinchilla's compute-optimal ratio (Hoffmann et al., 2022) is **D/N ≈ 20**. This run is not
undertrained by that standard — it's already ~25% _past_ compute-optimal for its size,
which was the deliberate target set in the scale-up spec (`N*≈244M`, `D*≈4.9B`, landed at
220.2M/5.51B).

`plots/chinchilla-comparison.png` puts this next to the _previous_ (75.5M-param) run, which
trained at 69 tokens/param — nearly 3.5x past Chinchilla-optimal, and which was _also_
diagnosed as capacity-limited (not undertrained) in the scale-up spec. The current run sits
much closer to the Chinchilla line than the previous one did, yet shows the same qualitative
symptom (weak factual recall).

## 3. The tell: matching flat-tail signatures across both runs

The scale-up decision itself was triggered by a specific empirical pattern in the _previous_
run's `val_loss` curve: a 0.588 drop in the first 500 steps vs. only a 0.005 drop in the last
500 of 10,000 — read as "deep in the flat tail of its own capacity-limited loss curve, not
merely undertrained" (`2026-08-09-model-scale-up-design.md`).

The current (220.2M-param) run's `val_loss` history shows the **same pattern, at nearly the
same magnitude**, despite being 3x bigger and much closer to Chinchilla-optimal:

|                                   | Δ val_loss, final 500 steps |
| --------------------------------- | --------------------------- |
| v0 (75.5M non-emb, 69 tok/param)  | 0.0050                      |
| v1 (220.2M non-emb, 25 tok/param) | 0.0052                      |

(right panel of `plots/chinchilla-comparison.png`). A bigger model, trained with a much
better token/param ratio, still flattens out to essentially the same late-stage rate of
improvement. That's the signature the project already used once to conclude "this
architecture has hit its ceiling" — it's reproducing here.

**Answer: the pretraining run is not undertrained relative to Chinchilla, and the SFT run
shows no overfitting signature either** (`val_loss` still gently declining at step 12,000,
no train/val gap). The bottleneck is model capacity at the current size — the same
conclusion, one scale step later. More `--max-steps` on this exact architecture would mostly
extend the flat tail, not fix factual recall like "capital of France." The next lever is
another Chinchilla-driven scale-up (bigger `N` _and_ proportionally more `D` together), sized
to whatever compute budget is available next.

## 4. Observability: weight and activation health

Beyond the loss curves, the checkpoints themselves were inspected directly to answer a
narrower question: is the "not enough capacity" verdict masking something else — e.g. dead
layers or dead neurons that mean the model isn't even using the 220.2M parameters it has?

**Method.** Both `my_checkpoints/step_10500.pt` (pretrain) and
`my_sft_checkpoints/step_12000.pt` (SFT) were loaded directly (no training code changes,
no checkpoint format changes). Two kinds of check:

- **Static weight stats** (no forward pass): Frobenius norm and mean-abs-weight per linear
  layer (`q_proj`, `kv_proj`, `out_proj`, `w_gate`, `w_up`, `w_down`), and per-layer
  `RMSNorm` gain mean/std, for all 20 blocks. The full weight-value distribution (not just
  its norm) is visualized directly from `state_dict` as a per-layer histogram heatmap —
  x-axis block index, y-axis weight value, color log-count — with ±1σ/±2σ traced on top
  per layer (`plots/weight-distribution-heatmap-pretrain.png`,
  `plots/weight-distribution-heatmap-sft.png`).
- **Forward-pass activation diagnostics**: forward hooks on `mlp.w_gate` and on each `Block`,
  run against a fixed batch of 8 short factual sentences (not a rigorous benchmark — just
  enough real text to get representative activations). Two metrics per layer, both standard
  in the layer-pruning/interpretability literature (e.g. Gromov et al., 2024, "The
  Unreasonable Ineffectiveness of the Deeper Layers"):
  - **Dead-neuron fraction**: for each of the `d_ff` SwiGLU gate neurons, whether its
    post-SiLU activation stays below `1e-3` in absolute value for _every_ token in the batch
    (the standard dead-ReLU-style criterion, adapted to SiLU).
  - **Residual-stream contribution**: cosine similarity between each block's input and
    output, and `‖output − input‖ / ‖input‖` — how much each block actually changes the
    residual stream vs. passing it through close to unchanged.

### Findings

1. **No dead neurons anywhere.** Dead-neuron fraction is exactly `0.0000` for all 20 layers
   in both checkpoints — every SwiGLU gate neuron fires on at least one token in the batch.
   The MLP capacity is being used; this isn't a ReLU-style collapse story.
2. **No layer has collapsed.** Weight Frobenius norms grow smoothly and monotonically with
   depth for both `attn.out_proj` and `mlp.w_down` (`plots/weight-norm-by-depth.png`), with
   no discontinuities or near-zero layers. Pretrain and SFT checkpoints are nearly
   superimposable on this plot — SFT didn't cause catastrophic forgetting or weight collapse
   at the parameter level, consistent with its clean loss curve (§1). `ln_f`'s gain shifted
   modestly from 1.78 (pretrain) to 1.92 (SFT) — a small, unremarkable sharpening, not a red
   flag.
3. **The distribution heatmap confirms (2) visually and adds one nuance.** Every weight
   matrix is a clean, unimodal, zero-centered distribution at every depth — no bimodal
   split, no spike-at-zero (the shape a genuinely dead sub-population of weights would
   leave), no fat outlier tail. The ±1σ/±2σ bands widen gradually and smoothly with depth
   for every matrix type, mirroring §4.2's norm-vs-depth finding from the weight-norm-only
   view. `attn.out_proj` shows the most visible growth (σ roughly triples from block 0 to
   block ~15) and a mild negative skew that grows with depth — the ±1σ band isn't
   symmetric around zero the way the other five matrix types are. Nothing here crosses into
   "dead" (a σ collapsing toward the bin width) at any depth in either checkpoint, and the
   pretrain/SFT heatmaps are visually indistinguishable, reinforcing point 2.
4. **Block 0 dominates; blocks 1–19 each do comparatively little** (`plots/dead-layer-diagnostic.png`).
   Block 0's input/output cosine similarity is ~0.14–0.16 — it substantially rewrites the
   residual stream, as expected for the block that converts raw token embeddings into
   contextual features. Every subsequent block sits at 0.89–0.97 cosine similarity, i.e. it
   nudges the stream rather than transforming it, and the contribution ratio (right panel)
   decays by roughly an order of magnitude from block 0 to blocks 1–19 and stays low
   (~0.2–1.0) through the rest of the stack. This is a known property of pre-norm residual
   transformers, not itself a bug — but it's a genuinely useful data point given the
   conclusion of §3 is "add more capacity": if the next scale-up leans on adding _depth_
   specifically, this same diagnostic should be re-run on the new checkpoint to confirm the
   added layers are contributing rather than becoming near-identity pass-throughs (the same
   pattern that motivates layer-pruning work on much larger models). It's not evidence to
   prefer width over depth on its own — just a check worth re-running rather than assuming
   depth scales for free.

**Net**: no dead-layer or dead-neuron problem. The weight/activation picture is healthy;
it corroborates rather than complicates the §3 conclusion — this model is using the capacity
it has, and simply has less of it than the task needs.

## 5. Recommended W&B metrics going forward

None of §4's checks currently run during training — they were one-off scripts against a
finished checkpoint. Cheapest-to-value first:

1. **`wandb.watch(model, log="all", log_freq=...)`.** W&B's built-in per-parameter weight and
   gradient histogram logging. This is close to free (a few lines in `train.py`'s
   `wandb.init` setup) and would have surfaced the weight-norm-by-depth picture in §4 live,
   across the whole run, instead of requiring a post-hoc script against the final checkpoint
   only. Start with `log_freq` at a multiple of `eval_interval` to keep payload size sane.
2. **Per-layer gradient norm, not just the global `grad_norm` already logged.** `train.py`
   currently logs one scalar (the norm returned by `clip_grad_norm_` over all parameters).
   Splitting this into a handful of groups — e.g. `grad_norm/block_0`, `grad_norm/block_19`,
   `grad_norm/embed` — would catch a layer-specific vanishing/exploding-gradient problem that
   a single aggregate norm can hide (a healthy global norm can average over one bad layer).
3. **Update-to-weight-norm ratio per layer** (`‖lr · grad‖ / ‖weight‖`, the diagnostic
   nanoGPT-style tooling popularized): healthy layers sit in a narrow band (roughly `1e-3`);
   outliers flag layers learning too fast or too slow relative to their own scale. Cheap to
   compute alongside the existing gradient-clipping step.
4. **Periodic dead-neuron / residual-contribution check**, i.e. §4's forward-hook diagnostic
   run automatically every N evals against a small fixed batch (reuse the existing validation
   batch — no new data dependency) rather than only by hand at the end. This is the one
   recommendation worth gating behind actually wanting it: it adds a forward pass with hooks
   at eval time, which is not free on a rented A100 clock, so only worth adding if depth
   keeps growing in future scale-ups and this diagnostic becomes a recurring question rather
   than a one-off.
5. **Deferred / not recommended yet: attention entropy per head.** Would show attention
   collapse (a head that always attends to one token). Not free — `F.scaled_dot_product_attention`
   is a fused kernel and doesn't expose attention weights, so this needs a periodic
   eager-mode forward pass on a fixed batch, purely for logging. Worth it only if a future
   diagnostic pass (like §4) turns up something that specifically points at attention rather
   than the MLP/residual-stream findings here — not worth the added complexity today, per
   this project's "don't add it until the problem shows up" principle.

Items 1–3 are the ones worth wiring into `train.py` now; they're cheap, always-on, and would
turn "did we have a dead layer" from a one-off forensic question back into something visible
on the dashboard during the _next_ run rather than something diagnosed after the fact.
