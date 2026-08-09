# Model Scale-Up + Documentation Consolidation — Design

Date: 2026-08-09

## Scope

The first pretraining run (100.7M total / 75.5M non-embedding params, `d_model=768,
n_layers=12, n_heads=12, n_kv_heads=4`, 10,000 steps / 5.24B tokens) and the SFT run
built on top of it (`smoltalk`, stopped manually at step 2000) both produced incoherent,
factually-wrong output (`generate.py` answered "capital of France" with "Strasbourg" and
produced non-sequiturs about days of the week). The pretraining `val_loss` curve
(pulled from W&B: `v0kw6t9i`, `daw15n2k`, `e4sc9ez0`) shows a textbook diminishing-returns
shape — a 0.588 drop in the first 500 steps vs. a 0.005 drop in the last 500 of 10,000 —
confirming the model is deep in the flat tail of its own capacity-limited loss curve, not
merely undertrained.

This spec covers raising `ModelConfig`/`TrainConfig` defaults to a size better matched to a
larger compute budget (~$50 in RunPod credits), plus consolidating the operational docs
(`docs/sft-quickstart.md` → merged into `docs/training-guide.md`) into a personal,
copy-paste-first guide reflecting the actual next run's exact values — no placeholders.

**In scope:** `ModelConfig` field defaults, `TrainConfig.max_steps` default, deleting
`docs/sft-quickstart.md` and rewriting the relevant parts of `docs/training-guide.md`,
fixing stale parameter-count claims in `README.md`, updating checkpoint-size references in
`CLAUDE.md`.

**Explicitly deferred / excluded:**

- No change to `lr`/`min_lr`/`warmup_steps`/`batch_size`/`gradient_accumulation_steps`/
  `beta1`/`beta2`/`weight_decay` — these don't need to scale with model size for this
  project, and the existing values already sit in the range LLM-pretraining practice uses
  for models in the new 220–250M-parameter range (see Components §1).
- No sequence-packing, `num_workers` tuning, or gradient checkpointing — same "revisit only
  if the problem shows up" status as `CLAUDE.md` already documents.
- No change to `DataConfig` (vocab size, tokenizer sample size, shuffle buffer, max_seq_len)
  — none of these are implicated by the capacity-vs-tokens finding below.
- No automated storage-capacity guard in code (e.g. checking free space before saving a
  checkpoint) — the fix here is operational (resize the network volume), not a new code
  path, consistent with this project's "don't add validation for scenarios that can't
  happen" principle once the volume is sized correctly.

## Motivation: why grow the model instead of just training longer

Using the standard training-FLOPs approximation `FLOPs ≈ 6·N·D` (N = non-embedding
params, D = tokens) and the project's own measured throughput (175K tokens/sec, i.e.
~79.3 TFLOPS achieved on the rented A100 at the current model size):

- Current run's total compute: `6 × 75.5M × 5.24B ≈ 2.37×10¹⁸ FLOPs`. This is already
  ~69 tokens/non-embedding-param (D/N) — well past the ~20 tokens/param Chinchilla-optimal
  ratio (Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models"), and
  already a deliberate choice per `docs/superpowers/specs/2026-08-06-pretraining-loop-hardening-design.md`
  (the "overtrain a small model for inference quality" approach LLaMA popularized).
- New budget (~25 GPU-hours reserved for pretraining out of ~31.4 total on-demand hours
  from $50 at $1.59/hr, after reserving ~10% restart buffer and ~2–3 hours for SFT): at the
  same achieved FLOPs/sec, `≈ 7.1×10¹⁸ FLOPs` — about 3x the old run's compute.
- Solving Chinchilla's compute-optimal frontier (`D/N ≈ 20`, so `FLOPs = 120·N²`) for that
  new budget: `N* ≈ 244M` non-embedding params, `D* ≈ 4.9B` tokens.

Conclusion: for this new compute budget, **the optimal use of the extra compute is a
bigger model, not more tokens on the same model** — the current architecture is already
past the point of getting much further out of additional data (the flattening curve is the
direct empirical evidence of this), so simply raising `--max-steps` on the same
`ModelConfig` (the "cheapest to implement" option originally considered) would mostly buy
more of that flat tail.

## Components

### 1. `ModelConfig` and `TrainConfig` changes

`src/llmtrain/training/config.py`:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 1024      # was 768
    n_layers: int = 20       # was 12
    n_heads: int = 16        # was 12
    n_kv_heads: int = 4      # unchanged
    dropout: float = 0.0
    rope_theta: float = 10000.0
```

Verified by direct instantiation (`TransformerLM(cfg)`, summing `.parameters()`):
**253,756,416 total params / 220,201,984 non-embedding params** (up from 100,682,496 /
75,516,672). This lands close to the `N* ≈ 244M` non-embedding optimum computed above.

```python
@dataclass
class TrainConfig:
    ...
    max_steps: int = 10500   # was 10000
```

At unchanged `batch_size=32`, `gradient_accumulation_steps=8`, `max_seq_len=2048`
(524,288 tokens/optimizer-step), this yields **~5.51B tokens** total — close to, and
deliberately a bit past, the `D* ≈ 4.9B` optimum, consistent with this project's
established overtraining philosophy rather than a change to it.

No other `TrainConfig`/`DataConfig`/`GenerationConfig` fields change. `n_kv_heads=4`
stays fixed (GQA group size grows from 3:1 to 4:1 query:kv heads, which is an unrelated,
acceptable side effect of raising `n_heads`, not a design goal).

### 2. Checkpoint size and network-volume capacity

Checkpoints store fp32 model weights + AdamW's two fp32 moment tensors: confirmed exactly
`12 bytes/param` against the observed old checkpoint size (100,682,496 × 12 =
1,208,189,952 bytes vs. the actual 1,208,316,995 — matches to within rounding). At the new
253,756,416 params, each checkpoint is **~3.05GB** (up from ~1.2GB).

`save_checkpoint()` writes the new `step_N.pt.tmp` file before `prune_old_checkpoints()`
deletes the oldest kept checkpoint, so with `keep_last_n_checkpoints=3` (unchanged
default) there is a brief window with 4 checkpoints resident simultaneously
(**~12.2GB**). This exceeds the current 10GB network volume outright, before even
counting the ~8.5GB of checkpoints already on it from the previous run.

Resolution (operational, not a code or config change): the network volume will be resized
to accommodate the new checkpoint size before the run starts (user action via the RunPod
console). Pretraining alone peaks at ~12.2GB, but Part 4's SFT runs afterward add their own
checkpoints on top of that while `step_10500.pt` must remain on the volume — realistic peak
usage across pretraining plus both SFT stages, if nothing is cleaned up in between, is closer
to ~27.4GB, so a resize to roughly 30GB (or cleaning up older pretraining checkpoints before
starting SFT) is the safer target, not 20GB. The old run's `checkpoints/step_10000.pt` and
its `tokenizer.json` have already been archived locally (verified via SHA-256 match against
the `resolve_local_path()` cache copy) and can be deleted from the volume to reclaim space
before the new run starts; `sft-checkpoints/` and `sft-checkpoints-smoltalk/` from the old
run are likewise safe to delete once confirmed no longer needed.

### 3. Documentation consolidation

- **Delete** `docs/sft-quickstart.md`.
- **Rewrite** the RunPod pretraining and SFT sections of `docs/training-guide.md` to be
  written for this specific user and this specific run, not a general audience:
  - Actual RunPod paths (`/workspace/checkpoints`, `~/llm-from-scratch` repo checkout).
  - Actual S3 API details for pulling checkpoints locally: bucket `304ulu3f96`, region
    `us-md-1`, endpoint `https://s3api-us-md-1.runpod.io` (already validated working in
    this session).
  - The pretraining command needs no architecture flags (`--d-model`, `--n-layers`, etc.)
    since they now come from the updated `ModelConfig` defaults — just `--dataset
    fineweb_edu`, `--checkpoint-dir`, `--wandb-project`.
  - Concrete, final step counts everywhere a value is needed — e.g. `--init-from-checkpoint
    /workspace/checkpoints/step_10500.pt` for the SFT stage, not a `step_<last-good>`
    placeholder — since the new `max_steps=10500` is now a fixed, known value.
  - Merge in the `.env` creation step and the S3-pull-for-evaluation workflow from the
    deleted `sft-quickstart.md`, updated with the bucket/region/endpoint above instead of
    generic placeholders.
  - Keep the existing local Mac smoke-test section (Part 1) largely as-is — it already
    deliberately overrides the architecture down to a tiny size for speed, which is
    unaffected by this change, though its "125M-parameter" caveat text needs updating to
    the new default.
- **Fix `README.md`**: the "~125M parameters" claim at the default config is stale even
  for the *old* defaults (verified default was 100.7M, not 125M) and needs updating to the
  new 253.8M/220.2M figures; same for the checkpoint-size caveat in the troubleshooting
  section.
- **Fix `CLAUDE.md`**: the `~1GB` checkpoint-size comment (in the `checkpoint.py`
  description) becomes `~3GB`; remove any reference to `docs/sft-quickstart.md` now that
  it's merged into `docs/training-guide.md`.

## Error handling

No new error paths. This is a config-default change plus documentation; existing
validation (`d_model % n_heads == 0`, `n_heads % n_kv_heads == 0`, `head_dim % 2 == 0`) is
satisfied by the new values (`1024 % 16 == 0`, `16 % 4 == 0`, `head_dim = 64`, `64 % 2 ==
0`).

## Testing strategy

No test changes required. Confirmed via `grep` that no test hardcodes the old
`d_model`/`n_layers`/`n_heads` values as expected defaults (`tests/test_config.py` only
asserts `n_kv_heads == 4`, which is unchanged, and exercises `ModelConfig` with explicit
overrides elsewhere). Existing `uv run pytest` should pass unmodified; run it after the
config change as a sanity check.

## Config changes summary

`ModelConfig.d_model` 768→1024, `ModelConfig.n_layers` 12→20, `ModelConfig.n_heads`
12→16 (`ModelConfig.n_kv_heads` unchanged at 4). `TrainConfig.max_steps` 10000→10500. No
new fields, no removed fields, no CLI surface changes (existing `--d-model`/`--n-layers`/
`--n-heads`/`--max-steps` flags already read their defaults from these dataclass fields).
