# Model Scale-Up + Documentation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise `ModelConfig`/`TrainConfig` defaults to the scaled-up architecture decided in `docs/superpowers/specs/2026-08-09-model-scale-up-design.md`, and consolidate `docs/sft-quickstart.md` into a personalized, copy-paste-first `docs/training-guide.md` with the new run's exact values.

**Architecture:** One dataclass edit (`src/llmtrain/training/config.py`), then three documentation files brought into sync with it and with each other: `README.md`, `CLAUDE.md`, and `docs/training-guide.md` (which absorbs `docs/sft-quickstart.md`, then deletes it).

**Tech Stack:** Python dataclasses, Markdown docs, `uv run pytest` for regression verification.

## Global Constraints

- New `ModelConfig` defaults: `d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4` — 253,756,416 total params / 220,201,984 non-embedding params (verified by direct instantiation in the design spec).
- New `TrainConfig.max_steps` default: `10500` (524,288 tokens/step × 10500 = 5,505,024,000 ≈ 5.51B tokens).
- New checkpoint size: exactly `12 bytes/param` → `253,756,416 × 12 = 3,045,077,000` bytes ≈ **3.05GB** per checkpoint (was ~1.2GB).
- The real pretraining run's final checkpoint will be `step_10500.pt` (10500 / `checkpoint_interval` default 125 = 84 exactly, so this lands on a checkpoint boundary) — use this literal filename everywhere in docs that reference "the pretraining checkpoint," never a `step_<last-good>`-style placeholder.
- Non-secret RunPod S3 identifiers that are safe to hardcode in committed docs: bucket `304ulu3f96`, region `us-md-1`, endpoint `https://s3api-us-md-1.runpod.io`. **Never** write actual `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` values (or `HF_TOKEN`/`WANDB_API_KEY`) into any committed file — those stay as `...` placeholders in `.env` examples, exactly as the existing docs already do.
- No `DataConfig`/`GenerationConfig` changes, no CLI surface changes (existing flags already read these dataclass fields as defaults).
- No new tests — confirmed in the design spec that no existing test hardcodes the old `d_model`/`n_layers`/`n_heads` values as expected defaults.

---

### Task 1: Bump `ModelConfig`/`TrainConfig` defaults

**Files:**
- Modify: `src/llmtrain/training/config.py:16-24` (`ModelConfig`), `:38` (`TrainConfig.max_steps`)

**Interfaces:**
- Produces: the new architecture/step-count constants that Tasks 2 and 3's documentation will cite verbatim (253,756,416 total params, 220,201,984 non-embedding params, `max_steps=10500`, ~5.51B tokens, ~3.05GB checkpoints).

- [ ] **Step 1: Edit `ModelConfig`**

In `src/llmtrain/training/config.py`, replace:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0
```

with:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 1024
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0
```

- [ ] **Step 2: Edit `TrainConfig.max_steps`**

In the same file, replace:

```python
    max_steps: int = 10000
```

with:

```python
    max_steps: int = 10500
```

(this is the only `TrainConfig` field changing — leave `batch_size`, `gradient_accumulation_steps`, `lr`, `min_lr`, `warmup_steps`, `weight_decay`, `beta1`, `beta2`, `checkpoint_interval`, `keep_last_n_checkpoints`, `eval_interval` untouched.)

- [ ] **Step 3: Verify param counts match the plan's Global Constraints**

Run:

```bash
uv run python -c "
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import ModelConfig
m = TransformerLM(ModelConfig())
n = sum(p.numel() for p in m.parameters())
print(n)
"
```

Expected output: `253756416`. If it doesn't match, re-check Step 1's values before continuing.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass (no test hardcodes the old architecture defaults — confirmed by `grep -rn "d_model=768\|n_layers=12\|n_heads=12" tests/` returning no matches before this change).

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/config.py
git commit -m "Scale up default model to 253.8M params, bump max_steps to 10500

Chinchilla-scaling analysis of the previous run's plateauing loss curve
(docs/superpowers/specs/2026-08-09-model-scale-up-design.md) showed the
old 100.7M-param architecture was already well past its own
compute-optimal token count; the new $50 compute budget is better spent
on a bigger model than more tokens on the same one."
```

---

### Task 2: Fix stale parameter-count references in `README.md` and `CLAUDE.md`

**Files:**
- Modify: `README.md:20`, `README.md:37-38`, `README.md:185-188`
- Modify: `CLAUDE.md:50-51`

**Interfaces:**
- Consumes: the exact param counts and checkpoint size from Task 1 (253,756,416 total / 220,201,984 non-embedding params, ~3.05GB checkpoints).

- [ ] **Step 1: Fix the GQA ratio line in `README.md`**

Replace:

```markdown
- **Grouped-query attention (GQA)**: separate `n_heads`/`n_kv_heads` (12/4 by default, a 3:1
  ratio) via `F.scaled_dot_product_attention(..., enable_gqa=True)`.
```

with:

```markdown
- **Grouped-query attention (GQA)**: separate `n_heads`/`n_kv_heads` (16/4 by default, a 4:1
  ratio) via `F.scaled_dot_product_attention(..., enable_gqa=True)`.
```

- [ ] **Step 2: Fix the default-config param-count line in `README.md`**

Replace:

```markdown
At the default config (`d_model=768`, `n_layers=12`, `n_heads=12`, `n_kv_heads=4`,
`vocab_size=32768`, `max_seq_len=2048`) the model is roughly 125M parameters.
```

with:

```markdown
At the default config (`d_model=1024`, `n_layers=20`, `n_heads=16`, `n_kv_heads=4`,
`vocab_size=32768`, `max_seq_len=2048`) the model is 253.8M parameters (220.2M
non-embedding).
```

- [ ] **Step 3: Fix the "Known limitations" checkpoint-size bullet in `README.md`**

Replace:

```markdown
- **Checkpoints are large even at smoke-test scale.** At the default 125M-parameter
  architecture, each `step_N.pt` is roughly 1.0GB (model + optimizer state), regardless of
  how few training steps produced it. `--keep-last-n-checkpoints` prunes old ones, but disk
  usage during a run should be planned for accordingly.
```

with:

```markdown
- **Checkpoints are large even at smoke-test scale.** At the default 253.8M-parameter
  architecture, each `step_N.pt` is roughly 3.05GB (model + optimizer state), regardless of
  how few training steps produced it. `--keep-last-n-checkpoints` prunes old ones, but disk
  usage during a run should be planned for accordingly.
```

- [ ] **Step 4: Fix the checkpoint-size comment in `CLAUDE.md`**

Replace:

```
                       # files (TrainConfig.keep_last_n_checkpoints, default 3) — checkpoints are ~1GB
                       # each at the real fineweb_edu-scale config, so unbounded accumulation over a
```

with:

```
                       # files (TrainConfig.keep_last_n_checkpoints, default 3) — checkpoints are ~3GB
                       # each at the real fineweb_edu-scale config, so unbounded accumulation over a
```

- [ ] **Step 5: Verify no stale figures remain**

Run: `grep -n "125M\|d_model=768\|n_layers=12\|n_heads=12\|(12/4" README.md CLAUDE.md`
Expected: no output (the one legitimate remaining `n_heads=12`-shaped string would be a CLI flag name like `--n-heads N`, which doesn't match this pattern, so a clean no-match confirms the fix is complete).

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Fix stale parameter-count and checkpoint-size references

README.md's '~125M parameters' claim was already stale for the old
architecture (verified default was 100.7M, not 125M) and both files
still described sizes for the pre-scale-up architecture."
```

---

### Task 3: Consolidate `docs/sft-quickstart.md` into `docs/training-guide.md`

**Files:**
- Modify: `docs/training-guide.md` (intro paragraph, Part 1 caveat, Part 3 rewrite, new Part 4, renumber old Part 4→5 and Part 5→6, all internal "Part N" cross-references, Part 6's checkpoint-cost bullet)
- Delete: `docs/sft-quickstart.md`
- Modify: `CLAUDE.md:60` (the one external "Part 5" cross-reference, which now needs to say "Part 6")

**Interfaces:**
- Consumes: Task 1's exact numbers (253,756,416 total / 220,201,984 non-embedding params, `max_steps=10500`, ~5.51B tokens, ~3.05GB checkpoints, `step_10500.pt` as the pretraining run's final checkpoint) and the already-validated RunPod S3 identifiers (bucket `304ulu3f96`, region `us-md-1`, endpoint `https://s3api-us-md-1.runpod.io`).

- [ ] **Step 1: Update the intro paragraph to mention SFT**

In `docs/training-guide.md`, replace:

```markdown
This is the full walkthrough for running this project end to end: a local smoke test on a
Mac, a real-scale smoke test on a rented A100, the actual `fineweb-edu` pretraining run, and
inference against a checkpoint with `generate.py`. See the repo root `README.md` for a
project overview; this doc assumes you've already read that and just want to run things.
```

with:

```markdown
This is the full walkthrough for running this project end to end: a local smoke test on a
Mac, a real-scale smoke test on a rented A100, the actual `fineweb-edu` pretraining run, SFT
on top of it, and inference against a checkpoint with `generate.py`. See the repo root
`README.md` for a project overview; this doc assumes you've already read that and just want
to run things.
```

- [ ] **Step 2: Fix the Part 1 architecture caveat**

Replace:

```markdown
**Important caveat about this run**: no `--d-model`/`--n-layers`/etc. override was passed, so
this ran the full default 125M-parameter architecture (`ModelConfig` defaults —
`d_model=768`, `n_layers=12`, `n_heads=12`, `n_kv_heads=4`). That's the architecture the real
pretraining run uses, which is exactly why it's a meaningful smoke test — but it's slow on
Mac CPU/MPS (tens of seconds per optimizer step) and produces roughly 1GB checkpoint files
even at 30 steps. If you just want to confirm the pipeline runs end to end without waiting or
burning disk, override the architecture down, e.g.:
```

with:

```markdown
**Important caveat about this run**: no `--d-model`/`--n-layers`/etc. override was passed, so
this ran the full default 253.8M-parameter architecture (`ModelConfig` defaults —
`d_model=1024`, `n_layers=20`, `n_heads=16`, `n_kv_heads=4`). That's the architecture the real
pretraining run uses, which is exactly why it's a meaningful smoke test — but it's slow on
Mac CPU/MPS (tens of seconds per optimizer step) and produces roughly 3GB checkpoint files
even at 30 steps. If you just want to confirm the pipeline runs end to end without waiting or
burning disk, override the architecture down, e.g.:
```

- [ ] **Step 3: Fix Part 2's forward references to the troubleshooting Part (5→6)**

Replace (line ~139):

```markdown
buffer and make `--resume` silently train zero steps (see Part 5). This is the dataset to use
```

with:

```markdown
buffer and make `--resume` silently train zero steps (see Part 6). This is the dataset to use
```

Replace (line ~245):

```markdown
— see Part 5's "dropped SSH connection" entry for why: without it, your SSH session dropping
```

with:

```markdown
— see Part 6's "dropped SSH connection" entry for why: without it, your SSH session dropping
```

Replace (line ~276):

```markdown
`CLAUDE.md`'s documented shuffle-buffer caveat (see Part 5), expect up to ~1000 rows to be
```

with:

```markdown
`CLAUDE.md`'s documented shuffle-buffer caveat (see Part 6), expect up to ~1000 rows to be
```

- [ ] **Step 4: Rewrite Part 3 (the real `fineweb_edu` pretraining run)**

Replace the entire block from `## Part 3 — The real \`fineweb_edu\` pretraining run` through
the end of its "Cost awareness: checkpoint storage" subsection (i.e. everything up to, but not
including, `## Part 4 — \`generate.py\` in depth`) with:

```markdown
## Part 3 — The real `fineweb_edu` pretraining run

Same code path as Parts 1 and 2 — `--dataset fineweb_edu` and a network-volume
`--checkpoint-dir`. No architecture flags are needed: `ModelConfig` defaults already encode
this run's architecture (`d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4` — 253.8M
total / 220.2M non-embedding parameters), and `TrainConfig.max_steps` already defaults to
`10500`.

```bash
nohup uv run --env-file .env python -m llmtrain.training.train --dataset fineweb_edu \
  --checkpoint-dir /workspace/checkpoints --wandb-mode online \
  > /root/train.log 2>&1 &
disown
```

(confirm your pod's network-volume mount with `df -h | grep workspace` first — swap
`/workspace` if yours differs.) This run lasts hours, not minutes, so `nohup ... & disown`
matters even more here — see Part 2 and Part 6's "dropped SSH connection" entry. The run's
final checkpoint will be `step_10500.pt` (10500 / `checkpoint_interval` (125) = 84 exactly,
so `max_steps` lands cleanly on a checkpoint boundary) — this is the exact filename Part 4's
SFT commands below point `--init-from-checkpoint` at.

At the defaults (`max_steps=10500`, `batch_size=32`, `gradient_accumulation_steps=8`,
`max_seq_len=2048`), this trains on `32 × 8 × 2048 = 524,288` tokens per optimizer step,
**~5.51B tokens total**. This model/token budget was picked via a Chinchilla-scaling
analysis of the previous run's plateauing loss curve (100.7M params, 5.24B tokens, val_loss
dropping 0.588 in its first 500 steps vs. only 0.005 in its last 500) — see
`docs/superpowers/specs/2026-08-09-model-scale-up-design.md` for the full reasoning. In
short: that run was already well past its own compute-optimal token count (~52
tokens/non-embedding-param vs. Chinchilla's ~20), so the extra compute this run spends goes
mostly into a bigger model (220.2M non-embedding params, up from 75.5M) with a token count
(~5.51B) close to *this* model's own ~4.9B-token optimum — deliberately a bit past it,
matching this project's established overtraining-for-inference-quality philosophy (same as
the original run, see
`docs/superpowers/specs/2026-08-06-pretraining-loop-hardening-design.md`).

Same `uv sync --extra cuda` prerequisite as Part 2 applies here (fused cross-entropy is on by
default and needs `liger-kernel`).

### Cost awareness: checkpoint storage — resize the network volume first

Checkpoints are exactly `12 bytes/param` (fp32 weights + AdamW's two fp32 moment tensors) —
confirmed against the previous run's checkpoint sizes. At 253,756,416 params, each
`step_N.pt` is **~3.05GB**, not the ~1.2GB of the previous architecture. `save_checkpoint()`
writes the new file before `prune_old_checkpoints()` deletes the oldest, so at the default
`--keep-last-n-checkpoints 3` there's a brief window with **4 checkpoints resident at once
(~12.2GB)**. Your network volume needs to be resized to comfortably clear that — **resize it
to at least 20GB** (RunPod console → Storage → your volume → resize; this generally requires
stopping the pod first) before launching the command above.

Free up space on the volume first by deleting the previous run's checkpoints — they're
already archived locally and verified (`~/Downloads/step_10000.pt` matches the SHA-256 of
the cached copy at `~/.cache/llmtrain/s3/304ulu3f96/checkpoints/step_10000.pt`), so this is
safe:

```bash
uv run --with boto3 --env-file .env python -c "
import boto3
s3 = boto3.client('s3')
bucket = '304ulu3f96'
for key in [
    'checkpoints/step_10000.pt', 'checkpoints/step_9875.pt', 'checkpoints/step_9750.pt',
    'sft-checkpoints/step_125.pt',
    'sft-checkpoints-smoltalk/step_1750.pt',
    'sft-checkpoints-smoltalk/step_1875.pt',
    'sft-checkpoints-smoltalk/step_2000.pt',
]:
    s3.delete_object(Bucket=bucket, Key=key)
    print('deleted', key)
"
```

(the small `tokenizer.json` objects in each of those prefixes don't need deleting — the new
pretraining/SFT runs overwrite them automatically at startup.)
```

- [ ] **Step 5: Insert the new Part 4 (SFT) between the old Part 3 and old Part 4**

Immediately before `## Part 4 — \`generate.py\` in depth`, insert:

```markdown
## Part 4 — SFT (`no_robots` sanity check, then `smoltalk`)

`--init-from-checkpoint` loads model weights only (no optimizer/step state) from a
pretraining checkpoint into a fresh SFT run — see `CLAUDE.md`'s architecture section for the
full `--init-from-checkpoint`/`--resume` distinction and footguns.

### 1. Sanity check on `no_robots`

Small, fast dataset — proves the SFT pipeline works before committing to the long `smoltalk`
run below. Points at the pretraining run's final checkpoint, `step_10500.pt`:

```bash
uv run --env-file .env python -m llmtrain.training.train \
  --dataset no_robots \
  --init-from-checkpoint /workspace/checkpoints/step_10500.pt \
  --checkpoint-dir /workspace/sft-checkpoints \
  --max-seq-len 2048 \
  --lr 3e-5 --min-lr 3e-6 --warmup-steps 20 \
  --max-steps 150 \
  --wandb-project llm-training
```

- Architecture flags don't need to be passed — `--init-from-checkpoint` auto-adopts them
  from the checkpoint's persisted `model_config`.
- `--lr`/`--min-lr` are ~10x lower than pretraining defaults (standard SFT practice);
  `--max-steps 150` is roughly 4 epochs over `no_robots` at the default effective batch size
  (256).
- `--max-seq-len` is **not** auto-adopted from the checkpoint — always pass it explicitly
  matching pretraining (`2048` here), or the two runs silently diverge.
- `--tokenizer-path` doesn't need to be passed — it defaults to `tokenizer.json` next to
  `--init-from-checkpoint`, which is where `train.py` always saves it.
- Once this completes cleanly (finite/decreasing `val_loss`, `generate.py` runs against the
  result), move on to `smoltalk` below.

### 2. Scale up: `smoltalk`

`--init-from-checkpoint` still points at the pretraining checkpoint, not the `no_robots`
output above — the sanity check only proves the pipeline works, it isn't a step to build on.
A separate `--checkpoint-dir` keeps the two SFT runs from colliding:

```bash
uv run --env-file .env python -m llmtrain.training.train \
  --dataset smoltalk \
  --init-from-checkpoint /workspace/checkpoints/step_10500.pt \
  --checkpoint-dir /workspace/sft-checkpoints-smoltalk \
  --max-seq-len 2048 \
  --lr 3e-5 --min-lr 3e-6 --warmup-steps 300 \
  --max-steps 12000 \
  --wandb-project llm-training
```

- `smoltalk`'s `all` config has ~1.0M train rows; at the default effective batch size (256),
  one epoch is ~3900 steps. `--max-steps 12000` (~3 epochs) is a starting point based on
  typical SFT recipes, not a value tuned against this specific model — watch `val_loss` on
  the W&B dashboard and stop the run manually (`Ctrl-C`) whenever it plateaus or you've seen
  enough, rather than treating 12000 as a number you must reach. `--checkpoint-interval`
  (default 125) means there's always a recent checkpoint to grab whenever you decide to stop.
- `--warmup-steps 300` scales up proportionally from the `no_robots` run's 20 (same warmup
  fraction of total steps).

### 3. Pull a checkpoint down for evaluation

RunPod exposes network volumes over an S3-compatible API, so you don't need `scp` or the pod
to be running — `generate.py` reads `s3://` paths directly (see Part 5 below for how this
works in general). Your bucket is the network volume's ID, `304ulu3f96`; the
endpoint/region (`https://s3api-us-md-1.runpod.io`, `us-md-1`) and your S3 API key pair are
already in your local `.env`.

Generate directly against the volume (downloads and caches under
`~/.cache/llmtrain/s3/304ulu3f96/`, so repeated runs against the same checkpoint don't
re-download it):

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint s3://304ulu3f96/sft-checkpoints-smoltalk/step_<N>.pt \
  --prompt "What is the capital of France?" \
  --max-new-tokens 200 --temperature 0.7 --repetition-penalty 1.2
```

(swap `sft-checkpoints-smoltalk` for `sft-checkpoints` to evaluate the `no_robots` sanity run
instead, and `step_<N>` for whichever step you stopped at.)

List what's actually on the volume if you're not sure which `step_N.pt` files survived
`--keep-last-n-checkpoints` pruning:

```bash
uv run --with boto3 --env-file .env python -c "
import boto3
s3 = boto3.client('s3')
for obj in s3.list_objects_v2(Bucket='304ulu3f96').get('Contents', []):
    print(obj['Key'], obj['Size'])
"
```

```

- [ ] **Step 6: Renumber old Part 4 to Part 5**

Replace:

```markdown
## Part 4 — `generate.py` in depth
```

with:

```markdown
## Part 5 — `generate.py` in depth
```

- [ ] **Step 7: Personalize the S3 example in the (now) Part 5**

Replace:

```markdown
**Both `--checkpoint` and `--tokenizer-path` accept an `s3://bucket/key` URI instead of a
local path** — useful for running `generate.py` straight against a pod's network volume after
the pod itself has stopped, without a manual `scp` first (RunPod exposes network volumes over
an S3-compatible API even when nothing is running). Requires the optional `s3` dependency
group (`uv sync --extra s3`) and, in `.env`, an S3 API key pair from the RunPod console plus
the endpoint/region:

```dotenv
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL_S3=https://s3api-<region>.runpod.io
AWS_DEFAULT_REGION=<region>
```

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint s3://<bucket>/checkpoints/step_10000.pt \
  --prompt "..."
```

`--tokenizer-path` doesn't need to be passed here either — it defaults to `tokenizer.json` in
the same S3 prefix as `--checkpoint`, same as the local-path default. First run downloads and
caches under `~/.cache/llmtrain/s3/<bucket>/<key>`; later runs against the same checkpoint skip
the download entirely (checkpoints are treated as immutable once written) — matters in
practice, since these are ~1GB+ files.
```

with:

```markdown
**Both `--checkpoint` and `--tokenizer-path` accept an `s3://bucket/key` URI instead of a
local path** — useful for running `generate.py` straight against a pod's network volume after
the pod itself has stopped, without a manual `scp` first (RunPod exposes network volumes over
an S3-compatible API even when nothing is running). Requires the optional `s3` dependency
group (`uv sync --extra s3`) and, in `.env`, an S3 API key pair from the RunPod console plus
the endpoint/region — already set up in this project's local `.env` (bucket `304ulu3f96`,
region `us-md-1`, endpoint `https://s3api-us-md-1.runpod.io`):

```dotenv
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL_S3=https://s3api-us-md-1.runpod.io
AWS_DEFAULT_REGION=us-md-1
```

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint s3://304ulu3f96/checkpoints/step_10500.pt \
  --prompt "..."
```

`--tokenizer-path` doesn't need to be passed here either — it defaults to `tokenizer.json` in
the same S3 prefix as `--checkpoint`, same as the local-path default. First run downloads and
caches under `~/.cache/llmtrain/s3/304ulu3f96/`; later runs against the same checkpoint skip
the download entirely (checkpoints are treated as immutable once written) — matters in
practice, since these are ~3GB+ files.
```

- [ ] **Step 8: Renumber old Part 5 to Part 6 and fix its internal reference**

Replace:

```markdown
## Part 5 — Troubleshooting and known limitations
```

with:

```markdown
## Part 6 — Troubleshooting and known limitations
```

Replace (the `Part 2 uses` sentence stays about Part 2, no change needed there — only the
checkpoint-cost bullet below needs a number fix):

```markdown
**Checkpoint storage is a real, capped cost, not just tidiness.** See Part 3 — ~1GB per
checkpoint at the default architecture, capped at `--keep-last-n-checkpoints` (default 3) via
`prune_old_checkpoints()`.
```

with:

```markdown
**Checkpoint storage is a real, capped cost, not just tidiness.** See Part 3 — ~3.05GB per
checkpoint at the default architecture, capped at `--keep-last-n-checkpoints` (default 3) via
`prune_old_checkpoints()`.
```

- [ ] **Step 9: Delete `docs/sft-quickstart.md`**

```bash
git rm docs/sft-quickstart.md
```

- [ ] **Step 10: Fix `CLAUDE.md`'s cross-reference to the renumbered Part 6**

In `CLAUDE.md`, replace:

```
                       # docs/training-guide.md Part 5
```

with:

```
                       # docs/training-guide.md Part 6
```

- [ ] **Step 11: Verify renumbering and cross-references are consistent**

Run:

```bash
grep -n "Part [0-9]" docs/training-guide.md
grep -n "Part [0-9]" CLAUDE.md
```

Expected: `docs/training-guide.md` shows headings in order 1, 2, 3, 4, 5, 6 with no
remaining reference to a stale "Part 5" meaning troubleshooting or "Part 4" meaning
`generate.py`; `CLAUDE.md` shows only "Part 6" (the troubleshooting/checkpoint-retry
reference) and "Part 1" (the smoke-test reference, unchanged). Also run:

```bash
grep -rn "sft-quickstart" . --include="*.md" 2>/dev/null
```

Expected: no output (file deleted, no dangling references).

- [ ] **Step 12: Commit**

```bash
git add docs/training-guide.md CLAUDE.md
git rm --cached docs/sft-quickstart.md 2>/dev/null || true
git commit -m "Consolidate sft-quickstart.md into training-guide.md, personalize for this run

Merges the SFT walkthrough into the main guide as Part 4, updates every
pretraining/SFT command to the scaled-up architecture's exact values
(step_10500.pt, the real S3 bucket/region/endpoint), and renumbers the
generate.py/troubleshooting parts accordingly."
```

## Self-Review Notes

- **Spec coverage:** Task 1 covers Component 1 (ModelConfig/TrainConfig defaults). Task 2
  covers Component 3's README.md/CLAUDE.md bullets. Task 3 covers Component 3's
  training-guide.md rewrite and sft-quickstart.md deletion, plus Component 2's storage
  guidance (volume resize, cleanup script) folded into Part 3's rewrite. All spec components
  have a task.
- **Placeholder scan:** The only remaining `<...>` markers left in the plan
  (`s3://.../step_<N>.pt` in Part 4 §3, and `AWS_ACCESS_KEY_ID=...`/`AWS_SECRET_ACCESS_KEY=...`
  in Part 5's `.env` example) are intentional — the former is a genuinely unknown
  future value (whichever step the user manually stops `smoltalk` training at), and the
  latter must never contain a real secret in a committed file. Every value that *is* known
  ahead of time (architecture, `max_steps`, checkpoint filenames, bucket/region/endpoint) is
  written out literally.
- **Type/value consistency:** `step_10500.pt` is used consistently across Part 3's own text
  and both Part 4 SFT commands' `--init-from-checkpoint` flags. The bucket ID `304ulu3f96`
  and endpoint `us-md-1`/`https://s3api-us-md-1.runpod.io` are identical everywhere they
  appear (Part 3's cleanup script, Part 4 §3, Part 5's rewritten example).
