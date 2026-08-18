# DPO pipeline commands

Checkpoint dir: `/workspace/sft-checkpoints-smoltalk/step_12000.pt` (confirm mount path with
`df -h | grep workspace` if `/workspace` doesn't exist).

## Smoke test (~20 prompts)

```bash
nohup uv run --env-file .env python -m llmtrain.generate_pairs \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_12000.pt \
  --output /workspace/dpo-pilot-pairs_raw.jsonl \
  --num-prompts 20 \
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

```bash
nohup uv run --env-file .env python -m llmtrain.training.dpo \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_12000.pt \
  --pairs /workspace/dpo-pilot-pairs_dpo.jsonl \
  --checkpoint-dir /workspace/dpo-pilot-checkpoints \
  --num-train-epochs 1 \
  > /root/dpo_train_pilot.log 2>&1 &
disown
```

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
