# Training guide

This is the full walkthrough for running this project end to end: a local smoke test on a
Mac, a real-scale smoke test on a rented A100, the actual `fineweb-edu` pretraining run, and
inference against a checkpoint with `generate.py`. See the repo root `README.md` for a
project overview; this doc assumes you've already read that and just want to run things.

Prerequisites for anything beyond `--wandb-mode disabled` local runs: a `.env` file at the
repo root (git-ignored) with `HF_TOKEN` and `WANDB_API_KEY` set, and `uv sync` run once to
install dependencies. Every command below is invoked as `uv run --env-file .env ...` so those
variables are available to the process; see `CLAUDE.md` for why (`HF_TOKEN`/`WANDB_API_KEY`
are required, never commit `.env`).

## Part 1 — Local smoke test (Mac, MPS/CPU)

This section was actually run end to end on this machine and the transcript below is real —
not paraphrased. Use it as your baseline "does the pipeline still work" check before spending
any money on a GPU rental.

```bash
uv run --env-file .env python -m llmtrain.training.train --dataset tiny_shakespeare \
  --max-steps 30 --gradient-accumulation-steps 2 --batch-size 4 --checkpoint-interval 10 \
  --eval-interval 10 --keep-last-n-checkpoints 2 --wandb-mode disabled \
  --checkpoint-dir /tmp/smoke-test --log-file /tmp/smoke-test.log
```

This completed successfully. `val_loss` at each eval step:

| step | val_loss |
|---|---|
| 10 | 8.0030 |
| 20 | 7.5120 |
| 30 | 7.2297 |

That decreasing trend is the real sanity signal to look for — not the absolute values, and
not (yet) the generated text quality; see below. Checkpoint pruning also behaved correctly:
with `--keep-last-n-checkpoints 2`, `step_10.pt` was deleted once `step_30.pt` was written,
leaving only `step_20.pt` and `step_30.pt` in `/tmp/smoke-test`.

**Important caveat about this run**: no `--d-model`/`--n-layers`/etc. override was passed, so
this ran the full default 125M-parameter architecture (`ModelConfig` defaults —
`d_model=768`, `n_layers=12`, `n_heads=12`, `n_kv_heads=4`). That's the architecture the real
pretraining run uses, which is exactly why it's a meaningful smoke test — but it's slow on
Mac CPU/MPS (tens of seconds per optimizer step) and produces roughly 1GB checkpoint files
even at 30 steps. If you just want to confirm the pipeline runs end to end without waiting or
burning disk, override the architecture down, e.g.:

```bash
uv run --env-file .env python -m llmtrain.training.train --dataset tiny_shakespeare \
  --d-model 64 --n-layers 2 --n-heads 2 --n-kv-heads 1 \
  --max-steps 30 --gradient-accumulation-steps 2 --batch-size 4 --checkpoint-interval 10 \
  --eval-interval 10 --keep-last-n-checkpoints 2 --wandb-mode disabled \
  --checkpoint-dir /tmp/smoke-test-small --log-file /tmp/smoke-test-small.log
```

This is much faster and produces small checkpoints, at the cost of being architecturally
unrepresentative of the real run — don't read anything into its loss curve or generated text
beyond "did the code path execute."

### Generating from the checkpoint

```bash
uv run python -m llmtrain.generate --checkpoint /tmp/smoke-test/step_30.pt \
  --prompt "Once upon a time" --max-new-tokens 20
```

This also ran successfully (real transcript, not paraphrased) and produced garbled but
well-formed text — no crash, no error. Garbled output at 30 steps is expected, not a bug:
`tiny_shakespeare` has only 472 train rows, 30 steps is nowhere near enough signal for
coherent text, and the default `temperature=1.0` with no `--top-k`/`--top-p` samples from the
full, largely-untrained distribution. If you want more legible-looking output to eyeball at
smoke-test scale, try tighter sampling:

```bash
uv run python -m llmtrain.generate --checkpoint /tmp/smoke-test/step_30.pt \
  --prompt "Once upon a time" --max-new-tokens 20 --temperature 0.7 --top-k 40
```

But treat this as a readability aid only — real judgment of model quality at any step count
should come from `val_loss` trending down (as it did above), not from reading smoke-test-scale
generated text, coherent-looking or not.

**Troubleshooting note**: earlier in this project's development, this exact `generate.py`
command would have crashed with `ValueError: loaded state dict has a different number of
parameter groups`. The cause was `generate.py` constructing a dummy optimizer purely to
satisfy `load_checkpoint()`'s signature, and that dummy optimizer's single param group didn't
match the two-param-group AdamW `train()` actually uses (decay vs. no-decay groups — see
`training/train.py`). This is fixed: `load_checkpoint()`'s `optimizer` parameter is now
optional (`src/llmtrain/training/checkpoint.py`), and `generate.py` no longer constructs one
at all. If you're on an older checkout and hit this, that's the fix to pull in.

## Part 2 — A100 smoke test (`reformer_enwik8`)

**This section has not been run against a real A100 in this session** — there's no CUDA
access in this environment. Unlike Part 1, treat what follows as a well-specified, ready-to-run
plan, not a verified transcript. Test it for real before trusting it blindly.

Per `CLAUDE.md`, `reformer_enwik8` (`reds0510/enwik8-processed` on the Hub, 1.1M rows) exists
specifically as a "~15-minute A100 smoke test... real-scale enough for `--resume` to behave
correctly" — unlike `tiny_shakespeare`, whose 472 rows are smaller than the default shuffle
buffer and make `--resume` silently train zero steps (see Part 5). This is the dataset to use
to actually exercise `--resume` before trusting it on the real `fineweb_edu` run.

### One-time setup on the pod

Pull these directly from `CLAUDE.md`'s RunPod workflow section — they're accurate and
already carefully worded:

- Rent **spot** instances — checkpoint-on-network-volume plus exact stream resume absorbs
  interruption risk, and spot is meaningfully cheaper.
- `--checkpoint-dir` points at a mounted **network volume** so checkpoints survive pod
  stop/restart independent of the pod's own disk.
- `pip install wandb && wandb login` once per pod; bake into a startup script/Dockerfile for
  custom templates so it survives fresh containers.
- `uv sync --extra cuda` once per pod to install `liger-kernel` — `TrainConfig.use_fused_ce`
  defaults to `True`, so training will `ImportError` partway into a run (after tokenizer
  training and dataset streaming) on a CUDA box that skips this step.
- Launch training inside **tmux** (or `nohup ... &`) and disconnect — monitor via the W&B
  dashboard, don't hold the SSH session open. No inbound port exposure is needed; W&B is
  outbound-only.

### Running the smoke test

The flag values below (`--max-steps`, `--checkpoint-interval`, `--eval-interval`) are a
reasonable starting point for a real-architecture ~15-minute run on an A100, chosen to land a
handful of checkpoints within that window — they are **not benchmarked against actual A100
throughput in this session**, so watch the first few step times in the log/W&B dashboard and
adjust `--max-steps` if 15 minutes runs long or short:

```bash
uv run --env-file .env python -m llmtrain.training.train --dataset reformer_enwik8 \
  --max-steps 150 --checkpoint-interval 25 --eval-interval 50 --keep-last-n-checkpoints 3 \
  --checkpoint-dir /workspace/network-volume/checkpoints --wandb-mode online
```

This uses the full default architecture and default `batch_size`/`gradient_accumulation_steps`
(no overrides), since the point of this smoke test is to be representative of the real run.
`model.compile()` will also kick in automatically here (gated on `device.type == "cuda"`), so
expect the very first optimizer step to be noticeably slower than the rest — that's
`torch.compile` warm-up, not a hang.

### Demonstrating `--resume`

Once the run above has written at least `step_50.pt` (interrupt the run after that point, or
just let it reach `--max-steps`), resume from that checkpoint and continue to the same
`--max-steps` target:

```bash
uv run --env-file .env python -m llmtrain.training.train --dataset reformer_enwik8 \
  --max-steps 150 --checkpoint-interval 25 --eval-interval 50 --keep-last-n-checkpoints 3 \
  --checkpoint-dir /workspace/network-volume/checkpoints \
  --resume /workspace/network-volume/checkpoints/step_50.pt --wandb-mode online
```

Confirm in the logs/W&B that the step counter picks up at 50 rather than restarting at 0, and
that `loss`/`val_loss` continue their existing trend rather than jumping back up to their
step-0 values — that's the signal `--resume` actually restored model/optimizer state and
dataset stream position rather than just resuming the step counter cosmetically. Per
`CLAUDE.md`'s documented shuffle-buffer caveat (see Part 5), expect up to ~1000 rows to be
silently skipped across the resume boundary — bounded and practically invisible against
1.1M rows, unlike on `tiny_shakespeare`.

## Part 3 — The real `fineweb_edu` pretraining run

Same code path as Parts 1 and 2 — the only required change is `--dataset fineweb_edu`, plus
real-scale `--max-steps` and a network-volume `--checkpoint-dir`:

```bash
uv run --env-file .env python -m llmtrain.training.train --dataset fineweb_edu \
  --checkpoint-dir /workspace/network-volume/checkpoints --wandb-mode online
```

Left at their `TrainConfig` defaults (`max_steps=10000`, `batch_size=32`,
`gradient_accumulation_steps=8`, `max_seq_len=2048`), this trains on `32 × 8 × 2048 =
524,288` tokens per optimizer step, `5.24B` tokens total — deliberately above the
~1.5B Chinchilla-optimal token count for this model's ~75.5M non-embedding parameters
(see `docs/superpowers/specs/2026-08-06-pretraining-loop-hardening-design.md`). That's an
intentional choice, not an oversight: it matches the "overtrain a small model for inference
quality" approach LLaMA popularized, trading extra training compute for a better model at a
fixed inference-time parameter count. No change to `--max-steps` is needed to get this
behavior — it falls out of the defaults.

Same `uv sync --extra cuda` prerequisite as Part 2 applies here (fused cross-entropy is on by
default and needs `liger-kernel`).

### Cost awareness: checkpoint storage

Checkpoints are confirmed ~1GB each at the default architecture. With the default
`--keep-last-n-checkpoints 3`, steady-state storage on the network volume is roughly 3GB —
this cap is deliberate: checkpoint pruning (`prune_old_checkpoints()` in
`training/checkpoint.py`) was added specifically because unbounded checkpoint accumulation
over a `max_steps=10000` run (potentially dozens of `checkpoint_interval`-spaced saves) was
flagged as a real network-volume cost risk, not just a tidiness concern. If you widen
`--keep-last-n-checkpoints` for extra resume safety margin, multiply accordingly — network
volumes are billed for their provisioned size regardless of how much of it is actually used,
so also right-size the volume itself up front rather than relying purely on the checkpoint
cap.

## Part 4 — `generate.py` in depth

```
python -m llmtrain.generate --checkpoint <path/to/step_N.pt> [--tokenizer-path PATH] \
    --prompt "..." [--max-new-tokens N] [--temperature F] [--repetition-penalty F] \
    [--top-k N] [--top-p F]
```

`--tokenizer-path` defaults to `tokenizer.json` next to the checkpoint (saved there by
`train.py`). Model architecture is reconstructed from the `model_config` persisted in the
checkpoint itself (falls back to `ModelConfig()` defaults for older checkpoints saved before
that field existed) — so architecture fields that change numerics without changing tensor
shapes (e.g. `rope_theta`) can't silently drift between training and generation.

Sampling happens per-token in `_sample()` (`src/llmtrain/generate.py`), in this exact order:

1. **Repetition penalty** (`--repetition-penalty`, default `1.0`) is applied first, and
   applies even to greedy decoding (`--temperature 0.0`) — it's what lets greedy avoid
   repetition loops, since without it argmax would happily repeat the same token forever.
   It uses the Keskar et al. (CTRL, 2019) formula, also used by Hugging Face `transformers`:
   for every token id already generated (this includes the prompt tokens, not just newly
   sampled ones), divide its logit by the penalty if the logit is positive, or multiply by
   the penalty if it's negative. Both directions push the token's probability down, so a
   single `penalty >= 1.0` value works regardless of the logit's sign. `penalty == 1.0` is a
   no-op and skips the computation entirely.
2. If `--temperature 0.0`, generation is greedy (`argmax`) and returns immediately —
   `--top-k`/`--top-p` are not applied in this branch.
3. Otherwise, logits are divided by `--temperature` (default `1.0`; lower values sharpen the
   distribution toward the most likely tokens, higher values flatten it).
4. **Top-k** (`--top-k`, default `0` = disabled) keeps only the `top_k` highest-logit tokens,
   setting every other logit to `-inf`.
5. **Top-p / nucleus sampling** (`--top-p`, default `1.0` = disabled) sorts logits descending,
   takes a cumulative softmax over them, and keeps the smallest prefix of tokens whose
   cumulative probability is `>= top_p` (the "first token that crosses the threshold" is kept,
   so the kept set's cumulative probability is always at least `top_p`, standard nucleus
   sampling behavior).
6. The remaining logits are softmaxed and sampled via `torch.multinomial`.

Generation itself is KV-cache-backed (`model/cache.py`'s `KVCache`): the full prompt is run
through the model once to seed the cache, then each subsequent token is generated from a
single-token forward pass rather than re-running the whole growing sequence.

## Part 5 — Troubleshooting and known limitations

**`--resume` silently drops examples across the resume boundary, catastrophically so on
`tiny_shakespeare`.** `datasets` (confirmed v5.0.1) doesn't preserve the shuffle buffer's
*contents* across `state_dict()`/`load_state_dict()` — only enough to resume the underlying
stream position. On `load_state_dict`, it refills the buffer by reading `buffer_size`
(default 1000) new elements from the stream before yielding again, and those refill elements
are never yielded themselves — so every `--resume` permanently drops up to `buffer_size`
examples. This is a property of `.shuffle()` itself, confirmed by isolating `.skip()` (which
round-trips exactly on its own) from `.shuffle()` (which loses `buffer_size` examples on its
own) in `tests/test_streaming.py::test_shuffled_skip_dataset_resumes_correctly_via_state_dict`
(marked `xfail`, not fixed). The practical impact differs by dataset:
- On `tiny_shakespeare` (472 rows, smaller than the 1000-row default shuffle buffer): the
  resumed stream comes up **completely empty**, and the run silently trains zero steps with
  no error raised. Not worth fixing for a dataset that only exists for a few-second local
  smoke test — but be aware of it if you ever test `--resume` against `tiny_shakespeare`
  specifically.
- On `reformer_enwik8`/`fineweb_edu` (1.1M and effectively-unbounded-streamed rows,
  respectively): the same mechanism drops ~1000 rows out of millions/billions per resume — a
  bounded, practically invisible loss, not a stream failure. This is why Part 2 uses
  `reformer_enwik8` to demonstrate `--resume`, not `tiny_shakespeare`.

**Checkpoint storage is a real, capped cost, not just tidiness.** See Part 3 — ~1GB per
checkpoint at the default architecture, capped at `--keep-last-n-checkpoints` (default 3) via
`prune_old_checkpoints()`.

**`--use-fused-ce` (default `True`) needs `liger-kernel`, and fails late, not immediately.**
On a CUDA box where `uv sync --extra cuda` was skipped, training gets through tokenizer
training and dataset streaming setup before hitting the `ImportError` on the first forward
pass — so don't assume a fast failure means it's a config problem elsewhere. Either run
`uv sync --extra cuda` or pass `--no-use-fused-ce` (falls back to the original full-logits
`next_token_loss()` path; also always the path used on MPS/CPU regardless of this flag, since
fused CE is gated on `device.type == "cuda"`).

**JSONL logs are noisier than they need to be.** The root logger is currently DEBUG with
`disable_existing_loggers: False` (`logging_config.py`), so `app.log` also fills with
third-party DEBUG noise (`httpx`, `datasets`, etc.) — confirmed in a real run. Harmless (still
valid JSONL, doesn't affect training) but noisy to read through; scoping `llmtrain`'s own
logger to DEBUG and root to WARNING would clean this up if it becomes a real problem.

**Older `generate.py` optimizer-shape crash.** Fixed — see the troubleshooting note in Part 1.
Only relevant if you're running `generate.py` from a checkout predating the fix.

## CLI flag reference

The full flag list for both entry points, from `--help` on the current codebase (each
default reads from `DataConfig`/`ModelConfig`/`TrainConfig`/`GenerationConfig` in
`training/config.py`, which is the single source of truth — run `--help` yourself to confirm
current defaults rather than trusting this list to stay in sync):

`train.py`:
```
--dataset {tiny_shakespeare,reformer_enwik8,fineweb_edu}
--shuffle-buffer-size, --max-seq-len, --tokenizer-vocab-size, --tokenizer-sample-size
--d-model, --n-layers, --n-heads, --n-kv-heads, --dropout, --rope-theta
--max-steps, --batch-size, --gradient-accumulation-steps, --grad-clip
--lr, --min-lr, --warmup-steps, --weight-decay, --beta1, --beta2, --seed
--checkpoint-dir, --checkpoint-interval, --keep-last-n-checkpoints, --eval-interval
--compile/--no-compile, --use-amp/--no-use-amp, --use-fused-ce/--no-use-fused-ce
--wandb-project, --wandb-mode {online,offline,disabled}, --log-file, --resume
```

`generate.py`:
```
--checkpoint (required), --tokenizer-path, --prompt (required)
--max-new-tokens, --temperature, --repetition-penalty, --top-k, --top-p
```
