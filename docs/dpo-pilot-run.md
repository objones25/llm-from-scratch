# DPO pipeline commands

Checkpoint dir: `/workspace/sft-checkpoints-smoltalk/step_12000.pt` (confirm mount path with
`df -h | grep workspace` if `/workspace` doesn't exist).

Run `uv sync && uv sync --extra cuda` first if you haven't already on this pod session --
`transformers`/`trl` are base dependencies this pipeline needs that a pod imaged before this
branch won't have installed.

## Smoke test (~20 prompts)

```bash
nohup uv run --env-file .env python -m llmtrain.generate_pairs \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_12000.pt \
  --output /workspace/dpo-pilot-pairs_raw.jsonl \
  --num-prompts 20 \
  --max-new-tokens 256 \
  > /root/dpo_generate_pairs_pilot.log 2>&1 &
disown
```

```bash
nohup uv run --env-file .env python -m llmtrain.judge \
  --input /workspace/dpo-pilot-pairs_raw.jsonl \
  --output /workspace/dpo-pilot-pairs_dpo.jsonl \
  > /root/dpo_judge_pilot.log 2>&1 &
disown
```

If the default judge (`--judge-provider together` with `meta-llama/Llama-3.3-70B-Instruct`) stops
honoring structured outputs, try `--judge-provider cerebras` with whatever strong instruct model
is currently live there (check the model's HF Hub page's "Inference Providers" listing, since
availability shifts), or fall back to `--judge-provider featherless-ai` with the original model.

```bash
nohup uv run --env-file .env python -m llmtrain.training.dpo \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_12000.pt \
  --pairs /workspace/dpo-pilot-pairs_dpo.jsonl \
  --checkpoint-dir /workspace/dpo-pilot-checkpoints \
  --num-train-epochs 1 \
  > /root/dpo_train_pilot.log 2>&1 &
disown
```

Note: the DPO-exported checkpoint's step number restarts from a small number (e.g. `step_2.pt`)
rather than continuing the SFT run's step count -- this is intentional (`trainer.state.global_step`
from the short DPO run, not the SFT step it started from), and is exactly why this runbook writes
to a separate `--checkpoint-dir` (`dpo-pilot-checkpoints`, `dpo-checkpoints`) instead of the SFT
checkpoint directory -- writing into the SFT dir would collide with its own `step_N.pt` files.

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint /workspace/dpo-pilot-checkpoints/step_N.pt \
  --prompt "What is the capital of France?"
```

## Full run (~2,500 prompts)

```bash
nohup uv run --env-file .env python -m llmtrain.generate_pairs \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_12000.pt \
  --output /workspace/dpo-pairs_raw.jsonl \
  --num-prompts 2500 \
  --max-new-tokens 256 \
  > /root/dpo_generate_pairs.log 2>&1 &
disown
```

```bash
nohup uv run --env-file .env python -m llmtrain.judge \
  --input /workspace/dpo-pairs_raw.jsonl \
  --output /workspace/dpo-pairs_dpo.jsonl \
  > /root/dpo_judge.log 2>&1 &
disown
```

```bash
nohup uv run --env-file .env python -m llmtrain.training.dpo \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_12000.pt \
  --pairs /workspace/dpo-pairs_dpo.jsonl \
  --checkpoint-dir /workspace/dpo-checkpoints \
  --num-train-epochs 1 \
  > /root/dpo_train.log 2>&1 &
disown
```

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint /workspace/dpo-checkpoints/step_N.pt \
  --prompt "What is the capital of France?"
```
