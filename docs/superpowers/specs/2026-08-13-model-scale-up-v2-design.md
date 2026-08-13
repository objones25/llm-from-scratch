# Model Scale-Up v2 — Design

Date: 2026-08-13

## Scope

`docs/pretrain-sft-scale-analysis.md` (2026-08-12) concluded that the current 220.2M-non-embedding-param
architecture (`d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4`, itself the result of the
first scale-up in `docs/superpowers/specs/2026-08-09-model-scale-up-design.md`) is still
capacity-limited, not undertrained: its `val_loss` flat-tail signature (0.0052 drop in the final
500 of 10,500 steps) reproduces the exact pattern that triggered the *first* scale-up, despite
already sitting at 25.0 tokens/non-embedding-param — past Chinchilla-optimal (20). More steps on
this architecture would mostly extend the flat tail rather than fix factual recall. This spec
covers a second `ModelConfig`/`TrainConfig` scale-up, sized against a new, larger compute budget,
plus the resulting checkpoint/network-volume and documentation updates.

**In scope:** `ModelConfig` field defaults, `TrainConfig.max_steps` default, network-volume
resize guidance, and updating `README.md`/`CLAUDE.md`/`docs/training-guide.md` wherever they
state the current 253.8M/220.2M parameter counts, ~3.05GB checkpoint size, `step_10500.pt`
filename, or the ~30GB volume-sizing figure from the first scale-up.

**Explicitly deferred / excluded** (same reasoning as the first scale-up spec, still applies at
this size):

- No change to `lr`/`min_lr`/`warmup_steps`/`batch_size`/`gradient_accumulation_steps`/
  `beta1`/`beta2`/`weight_decay` — nothing about doubling `d_model`/params inherently requires
  retuning these for a project at this scale, and revisiting them isn't informed by anything in
  the scale analysis doc (which found no optimization problem — smooth, monotonic loss curves).
  If the next run's loss curve shows instability at the new size, that's a separate, later
  investigation.
- No sequence-packing, `num_workers` tuning, or gradient checkpointing — unchanged "revisit only
  if the problem shows up" status.
- No change to `DataConfig` (vocab size, tokenizer sample size, shuffle buffer, max_seq_len).
- No automated storage-capacity guard in code — the fix is operational (resize the network
  volume), consistent with this project's "don't add validation for scenarios that can't happen"
  principle.
- No change to the W&B observability recommendations in `docs/pretrain-sft-scale-analysis.md`
  §5 — those are tracked separately and aren't blocking this scale-up.

## Motivation: sizing against the new compute budget

Same method as the first scale-up spec: `FLOPs ≈ 6·N·D` (N = non-embedding params, D = tokens),
solved against Chinchilla's compute-optimal frontier (`D/N ≈ 20` ⟹ `FLOPs = 120·N²`), using the
project's measured achieved throughput of **79.3 TFLOPs/sec** on the rented A100 (established in
the first scale-up spec from 175K tokens/sec at the original 75.5M-param architecture; treated as
a roughly constant hardware-utilization rate independent of model size, the same assumption the
first scale-up spec made).

Budget: **~$160** in RunPod credits (up from ~$50 the first time), at the same $1.59/hr on-demand
A100 spot-equivalent pricing basis:

- Total on-demand hours: `160 / 1.59 ≈ 100.6`.
- Reserve ~10% restart buffer and ~2.5 hours for the SFT stage (same fixed SFT reserve as the
  first scale-up spec used): **~87.9 GPU-hours for pretraining**.
- FLOPs budget: `79.3×10¹² × 87.9 × 3600 ≈ 2.51×10¹⁹`.
- Chinchilla-solving for `N`: `N* ≈ √(2.51×10¹⁹ / 120) ≈ 457M` non-embedding params.

A literal doubling of the current 220.2M (→440.4M) sits close to this optimum, which is why
"double it" is directionally the right instinct — but the exact number needs to land on a valid,
efficient architecture (divisibility constraints, a sane head dimension), not the literal 2.000x
figure.

### Landing on a concrete architecture

`d_model` must be divisible by `n_heads`, `n_heads` by `n_kv_heads`, and `head_dim` (`d_model /
n_heads`) must be even (RoPE) — and, not enforced by code but a real efficiency concern, `head_dim`
should be a "nice" hardware-friendly value (a multiple of 8, ideally 64) rather than an arbitrary
even number, since the 79.3 TFLOPs/sec baseline this whole calculation rests on was measured at
`head_dim=64` and isn't guaranteed to hold at an unusual shape (e.g. `head_dim=46` was rejected for
exactly this reason during design).

Searching the valid grid at `n_layers=20` (unchanged depth — see next section for why) for the
best fit near 440–460M non-embedding params, with `head_dim` a multiple of 8:

| d_model | n_layers | n_heads | head_dim | n_kv_heads | non-emb params | total params |
| ------- | -------- | ------- | -------- | ---------- | --------------- | ------------- |
| 1440    | 20       | 20      | 72       | 4          | **431.4M**       | 478.6M        |

Verified by direct instantiation (`TransformerLM(cfg)`, summing `.parameters()`, same method the
first scale-up spec used): **478,553,760 total / 431,367,840 non-embedding** — 1.96x the current
220.2M non-embedding params.

### Why depth stays at 20 layers

`docs/pretrain-sft-scale-analysis.md` §4.4 found block 0 substantially rewrites the residual
stream (cosine similarity ~0.14–0.16 between its input/output) while blocks 1–19 sit at 0.89–0.97
— each nudges the stream rather than transforming it, with contribution ratio dropping roughly an
order of magnitude after block 0. That doc explicitly flagged this as a data point against
blindly adding *depth* in the next scale-up: more layers risk becoming additional near-identity
pass-throughs rather than adding usable capacity, the same pattern that motivates layer-pruning
work on larger models. This scale-up grows only width (`d_model` 1024→1440, `n_heads` 16→20,
GQA ratio unchanged at 5:1 relative to `n_kv_heads=4`) and holds depth fixed, avoiding that risk
entirely rather than gambling on it. If a future scale-up wants to grow depth, §4.4's diagnostic
should be re-run against this checkpoint first to confirm blocks 1–19 are pulling their weight
before adding more of them.

### Token budget and overtraining margin

```python
TrainConfig.max_steps: int = 18500   # was 10500
```

At unchanged `batch_size=32`, `gradient_accumulation_steps=8`, `max_seq_len=2048` (524,288
tokens/optimizer-step), this yields **9.70B tokens total** — 22.5 tokens/non-embedding-param,
past Chinchilla-optimal (20) and in the same overtraining range as both prior runs (69 and 25.0
tokens/param respectively), consistent with this project's established
overtrain-for-inference-quality philosophy. `18500` divides evenly by both `checkpoint_interval`
(125 → 148 checkpoints) and `eval_interval` (500 → 37 evals), so `max_steps` lands cleanly on a
checkpoint boundary exactly as `10500` did before — the final checkpoint will be `step_18500.pt`.

Estimated cost to run this to completion: ~87.9 GPU-hours pretraining + ~2.5 hours SFT + ~10%
restart buffer ≈ **~$160** at $1.59/hr — matches the budget this was sized against, by
construction.

## Components

### 1. `ModelConfig` and `TrainConfig` changes

`src/llmtrain/training/config.py`:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 1440      # was 1024
    n_layers: int = 20       # unchanged
    n_heads: int = 20        # was 16
    n_kv_heads: int = 4      # unchanged
    dropout: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    ...
    max_steps: int = 18500   # was 10500
```

No other `TrainConfig`/`DataConfig`/`GenerationConfig` fields change. `n_kv_heads=4` stays fixed
(GQA group size stays 5:1 query:kv heads, unchanged from the current architecture).

### 2. Checkpoint size and network-volume capacity

Checkpoints are `12 bytes/param` (fp32 weights + AdamW's two fp32 moment tensors, confirmed
exactly against both prior runs' checkpoint sizes). At 478,553,760 params, each `step_N.pt` is
**~5.74GB** (up from ~3.05GB).

Using the same peak-usage formula the first scale-up spec derived and the actual run's operations
confirmed (pretraining transient peak at `keep_last_n_checkpoints=3` [4 checkpoints resident
during the save-then-prune window] + one retained final pretraining checkpoint held through the
SFT stages + the `smoltalk` SFT run's own transient peak — the `no_robots` sanity-check stage
contributes only its own small transient overlap on top, folded into the same "+1 retained
checkpoint" term for a conservative estimate): **~9× checkpoint size ≈ 51.7GB** peak usage across
pretraining plus both SFT stages if nothing is cleaned up in between.

Resolution (operational, not a code change, same pattern as the first scale-up): resize the
network volume to **at least 60GB** before starting the pretraining run (RunPod console →
Storage → volume → resize; requires stopping the pod first), or plan to delete older pretraining
checkpoints before starting the `smoltalk` SFT stage if you'd rather stay closer to the ~52GB
realistic peak. The existing `step_10500.pt`/`step_12000.pt` (SFT) checkpoints and `tokenizer.json`
from the current run should be archived locally (matching the first scale-up's verified-via-SHA-256
pattern) and deleted from the volume before the new run starts, to reclaim space.

### 3. Documentation updates

- **`README.md`**: update the "253.8M parameters (220.2M non-embedding)" line (§ Architecture
  overview) to the new 478.6M/431.4M figures; update the "~3.05GB" checkpoint-size line in
  §Known limitations to ~5.74GB.
- **`CLAUDE.md`**: update the `~3GB`/`~3GB+` checkpoint-size references in the `checkpoint.py`
  and `s3.py` description comments to `~5.74GB`.
- **`docs/training-guide.md`**:
  - Part 3: update the "`d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4` — 253.8M total /
    220.2M non-embedding parameters" and "`max_steps` already defaults to `10500`" text to the
    new architecture and step count; update the token-budget paragraph (~5.51B tokens →
    ~9.70B tokens, 25.0 → 22.5 tokens/param) and its citation to point at this spec instead of
    (or alongside) the first scale-up spec; update `step_10500.pt` references to `step_18500.pt`
    throughout Parts 3–5 (including the `--init-from-checkpoint` commands in Part 4).
  - "Cost awareness: checkpoint storage" section: update the `12 bytes/param` math to the new
    param count (~5.74GB/checkpoint), the "~12.2GB" pretraining-peak figure, the "~27.4GB"
    combined-peak figure, and the "resize it to at least 30GB" guidance to the new ~51.7GB/60GB
    figures derived above.
  - The checkpoint-deletion S3 script (cleaning up the old run before starting the new one):
    update the hardcoded keys (`step_10000.pt`, `step_9875.pt`, `step_9750.pt`,
    `sft-checkpoints/step_125.pt`, `sft-checkpoints-smoltalk/step_{1750,1875,2000}.pt`) to the
    actual keys present from the *current* (220.2M) run being retired — `step_10500.pt` and its
    two preceding checkpoints, plus the current run's SFT checkpoints, all still to be
    determined at execution time from what's actually on the volume, not guessed here.

## Error handling

No new error paths. This is a config-default change plus documentation; existing validation
(`d_model % n_heads == 0`, `n_heads % n_kv_heads == 0`, `head_dim % 2 == 0`) is satisfied by the
new values (`1440 % 20 == 0`, `20 % 4 == 0`, `head_dim = 72`, `72 % 2 == 0`).

## Testing strategy

No test changes required. Confirmed via `grep` that no test hardcodes the current
`d_model`/`n_layers`/`n_heads` values as expected defaults — `tests/test_config.py` only asserts
`n_kv_heads == 4` (unchanged) and exercises `ModelConfig` with explicit small overrides
(`d_model=64, n_layers=4`) elsewhere. Existing `uv run pytest` should pass unmodified; run it
after the config change as a sanity check.

## Config changes summary

`ModelConfig.d_model` 1024→1440, `ModelConfig.n_heads` 16→20 (`ModelConfig.n_layers` and
`ModelConfig.n_kv_heads` unchanged at 20 and 4). `TrainConfig.max_steps` 10500→18500. No new
fields, no removed fields, no CLI surface changes (existing `--d-model`/`--n-heads`/`--max-steps`
flags already read their defaults from these dataclass fields).
