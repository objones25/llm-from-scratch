# Running the local smoke test

This is the walking skeleton's acceptance test: proves the full pipeline
(streaming data → tokenizer → model → training loop → checkpointing →
logging → W&B) runs end-to-end on real data, on your Mac, before any money
is spent on a rented GPU. It's a manual step — it needs your actual W&B
login, so it isn't automated.

## 1. Set up credentials

Create (or edit) `.env` at the repo root with:

```
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

- `HF_TOKEN`: not strictly required for `tiny_shakespeare` (it's a public
  dataset), but needed later for gated/private datasets.
- `WANDB_API_KEY`: get it from https://wandb.ai/settings#apikeys — it's a
  ~40-character key with no prefix (not the same as anything you'd paste
  into a URL or a `wandb login` prompt).

Copy `.env` into whichever checkout you're running from (main branch or a
worktree) — it's git-ignored and never committed.

## 2. Run every command with `uv run`

`wandb`, `python`, etc. are installed into this project's uv-managed
virtual environment, not your system Python. Bare `wandb login` or
`python ...` will fail with "command not found" or import errors — always
prefix with `uv run`, from the project root (where `pyproject.toml` lives).

## 3. Verify W&B auth

```bash
uv run --env-file .env wandb login --verify
```

Expected output: `Currently logged in as: <your username>`. If you see
`AuthenticationError` or `No API key configured`, the key in `.env` is
missing, mistyped, or wrong — re-check
https://wandb.ai/settings#apikeys and fix `.env`, then retry.

## 4. Run the smoke test

`config.py`'s baked-in defaults now target the real `fineweb_edu`
pretraining run (~101M params, `d_model=768`, `n_layers=12`,
`max_seq_len=2048`) — far too large for a quick local check. The CLI
exposes every `DataConfig`/`ModelConfig`/`TrainConfig` field as a flag
(see `--help`), so the smoke test overrides architecture down to the
original toy scale on top of the dataset/step/checkpoint overrides:

```bash
uv run --env-file .env python -m llmtrain.training.train \
    --dataset tiny_shakespeare --max-steps 50 --batch-size 4 --checkpoint-interval 50 \
    --d-model 128 --n-layers 2 --n-heads 4 --n-kv-heads 2 \
    --max-seq-len 128 --tokenizer-vocab-size 1000 --warmup-steps 5
```

This streams `Trelis/tiny-shakespeare`, trains a tiny BPE tokenizer on a
sample of it, builds a tiny transformer (2 layers, `d_model=128`), and runs
50 training steps on CPU/MPS, logging to W&B and to `app.log` (JSONL).
`--checkpoint-interval 50` overrides `TrainConfig`'s production default of
1000 so this quick run actually produces a checkpoint (see step 4) instead
of finishing before the first checkpoint interval. `--warmup-steps 5`
overrides `TrainConfig`'s production default of 200 — at the production
default, `get_lr`'s linear warmup never gets past ~25% of `--lr` within
only 50 steps, and the loss barely moves; scaling warmup down with the run
length keeps the LR schedule actually engaged.

## 5. Check all four success criteria

1. **Process exits 0, no traceback.**
2. **`app.log` is valid JSONL** — every line parses as JSON:
   ```bash
   uv run python -c "import json; [json.loads(l) for l in open('app.log')]"
   ```
   (no output / no exception = pass)
3. **W&B loss trends downward** — open the run URL printed to stdout,
   confirm the loss curve decreases over the 50 steps.
4. **A checkpoint was written:**
   ```bash
   ls checkpoints/
   ```
   should show at least one `step_*.pt` file.

If any check fails, it's a bug in one of the underlying modules (data,
tokenizer, model, checkpoint, or logging) — not in this runbook. Add or
adjust a unit test that would have caught it, fix the module, and re-run
from step 4.

## 5b. Generate text from the checkpoint

Confirms the inference path (`generate.py`, KV-cache decoding) also works
end-to-end against a checkpoint produced by step 4, using the tokenizer
saved alongside it:

```bash
uv run python -m llmtrain.generate --checkpoint checkpoints/step_50.pt \
    --prompt "Once upon a time" --max-new-tokens 20
```

Expected: prints generated text to stdout with no traceback. The output
won't be coherent (tiny model, tiny smoke-test training run) — this step
only verifies the generation pipeline runs, not output quality.

## 6. Record the result

Once all four checks pass:

```bash
git commit --allow-empty -m "chore: confirm tiny_shakespeare smoke test passes end-to-end"
```

## 7. Larger test run on reformer_enwik8

`ModelConfig`/`TrainConfig`/`DataConfig` now default to this run (~101M
params, `d_model=768`, `n_layers=12`, `n_heads=12`, `n_kv_heads=4`,
`vocab_size=32768`, `max_seq_len=2048`, `batch_size=32`, `max_steps=10000`)
against `reds0510/enwik8-processed`. This is A100 scale, not a quick local
check — run it on a rented pod, not your Mac:

```bash
uv run --env-file .env python -m llmtrain.training.train \
    --dataset reformer_enwik8
```

No flags needed beyond `--dataset` since the defaults already match. Same
four checks as step 5 apply; checkpoints land every 1000 steps.
