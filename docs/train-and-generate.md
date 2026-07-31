# Multi-epoch run on tiny_shakespeare, then generate

## 1. Train

`tiny_shakespeare` has 472 train rows. At `--batch-size 4`, one epoch is
`ceil(472 / 4) = 118` steps. `train()` re-iterates the dataset whenever the
stream runs dry before `--max-steps` is reached, so `--max-steps 150` runs
past one epoch into a second, checkpointing every `checkpoint_interval`
(default 50) along the way — `step_50.pt`, `step_100.pt`, `step_150.pt`.

```bash
uv run --env-file .env python -m llmtrain.training.train \
    --dataset tiny_shakespeare --max-steps 150 --batch-size 4 --checkpoint-dir checkpoints
```

Requires `WANDB_API_KEY` in `.env` (see `docs/smoke-test.md` for setup).
The tokenizer is saved alongside the checkpoints at `checkpoints/tokenizer.json`.

150 steps still isn't enough to produce coherent text — `tiny_shakespeare`
exists to smoke-test the pipeline end-to-end (per `CLAUDE.md`), not to
train a good model; expect degenerate/repetitive output at this scale
regardless of epoch count. Push `--max-steps` well past 150 (e.g. several
thousand) if you want to see the loss curve actually converge somewhere
meaningful.

## 2. Generate

```bash
uv run python -m llmtrain.generate \
    --checkpoint checkpoints/step_150.pt \
    --prompt "To be, or not to be" --max-new-tokens 50 --temperature 0.8
```

`--tokenizer-path` defaults to `tokenizer.json` next to the checkpoint, so
it doesn't need to be passed explicitly here. No `.env` needed — generation
is local-only, no W&B or Hugging Face network access required.
