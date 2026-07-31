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

```bash
uv run --env-file .env python -m llmtrain.training.train \
    --dataset tiny_shakespeare --max-steps 50 --batch-size 4
```

This streams `karpathy/tiny_shakespeare`, trains a tiny BPE tokenizer on a
sample of it, builds the minimal transformer, and runs 50 training steps
on CPU/MPS, logging to W&B and to `app.log` (JSONL).

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

## 6. Record the result

Once all four checks pass:

```bash
git commit --allow-empty -m "chore: confirm tiny_shakespeare smoke test passes end-to-end"
```
