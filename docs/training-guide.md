# Training guide

Commands to set up this repo on a rented RunPod pod and run the full pipeline in order:
pretraining → SFT → DPO. For local Mac smoke testing see `README.md`; for full technical
detail (flags, architecture, known limitations) see `CLAUDE.md`; for past-run results see
`docs/dpo-run-results.md` and `docs/pretrain-sft-scale-analysis.md`.

Every long-running command below is launched with `nohup ... & disown` — this survives an
SSH drop, a laptop going to sleep, or a closed terminal. Don't run these in the foreground.

## Pod setup

```bash
# fresh pod only -- official RunPod PyTorch images don't ship uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

git clone https://github.com/objones25/llm-from-scratch.git && cd llm-from-scratch
uv sync --extra cuda   # add --extra s3 too if you'll pull checkpoints via s3:// paths

cat > .env << 'EOF'
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
EOF
uv run --env-file .env wandb login --verify
```

Confirm the network volume's mount path before pointing `--checkpoint-dir` at it — don't
assume `/workspace`:

```bash
df -h | grep workspace
```

## 1. Pretraining (`fineweb_edu`)

```bash
nohup uv run --env-file .env python -m llmtrain.training.train --dataset fineweb_edu \
  --checkpoint-dir /workspace/checkpoints --wandb-mode online \
  > /root/train.log 2>&1 &
disown
```

If it dies (pod crash, OOM, `kill`), resume from the last checkpoint (`tail /root/train.log`
or the W&B dashboard to find `step_<N>`):

```bash
nohup uv run --env-file .env python -m llmtrain.training.train --dataset fineweb_edu \
  --checkpoint-dir /workspace/checkpoints --wandb-mode online \
  --resume /workspace/checkpoints/step_<N>.pt \
  > /root/train_resume.log 2>&1 &
disown
```

Default `--max-steps` (18500) lands the final checkpoint at `step_18500.pt` — that's what
Part 2 below points `--init-from-checkpoint` at.

## 2. SFT

### Sanity check on `no_robots` first (fast, ~150 steps)

```bash
nohup uv run --env-file .env python -m llmtrain.training.train --dataset no_robots \
  --init-from-checkpoint /workspace/checkpoints/step_18500.pt \
  --checkpoint-dir /workspace/sft-checkpoints \
  --max-seq-len 2048 --lr 3e-5 --min-lr 3e-6 --warmup-steps 20 --max-steps 150 \
  --wandb-project llm-training \
  > /root/train_sft_no_robots.log 2>&1 &
disown
```

Confirm it finishes cleanly (decreasing `val_loss`, `generate.py --chat` runs against the
result) before moving on.

### Then `smoltalk` (the real SFT run)

```bash
nohup uv run --env-file .env python -m llmtrain.training.train --dataset smoltalk \
  --init-from-checkpoint /workspace/checkpoints/step_18500.pt \
  --checkpoint-dir /workspace/sft-checkpoints-smoltalk \
  --max-seq-len 2048 --lr 3e-5 --min-lr 3e-6 --warmup-steps 300 --max-steps 12000 \
  --wandb-project llm-training \
  > /root/train_sft_smoltalk.log 2>&1 &
disown
```

Resume if interrupted (`--init-from-checkpoint` still points at the pretraining checkpoint
for a *fresh* SFT run — this is for continuing this exact interrupted run):

```bash
nohup uv run --env-file .env python -m llmtrain.training.train --dataset smoltalk \
  --checkpoint-dir /workspace/sft-checkpoints-smoltalk \
  --resume /workspace/sft-checkpoints-smoltalk/step_<N>.pt --wandb-project llm-training \
  > /root/train_sft_resume.log 2>&1 &
disown
```

`--resume` rebuilds the model from CLI/default `ModelConfig` flags, not the checkpoint's own
saved architecture — if the original `--init-from-checkpoint` run used non-default
`--d-model`/`--n-layers`/etc., pass the same flags here too, or `load_state_dict` throws a
shape-mismatch error (see `CLAUDE.md`).

To stop a `smoltalk` run early (it's long — watch W&B and stop whenever `val_loss` plateaus):

```bash
pgrep -f 'dataset smoltalk' && kill <pid>   # not Ctrl-C, this runs detached
```

### Evaluate a checkpoint

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_<N>.pt \
  --chat --prompt "What is the capital of France?" \
  --max-new-tokens 200 --temperature 0.7 --repetition-penalty 1.2
```

`--chat` is required for this (and every) SFT/DPO checkpoint — omitting it feeds the model a
prompt it never saw at position 0 during training and produces garbled output that looks
like a model-quality problem but isn't (see `docs/dpo-run-results.md` §4).

## 3. DPO (preference tuning)

Three stages: sample completion pairs from the SFT checkpoint, judge them into
chosen/rejected preference pairs, then train with TRL's `DPOTrainer`.

### 3a. Sample completion pairs

```bash
nohup uv run --env-file .env python -m llmtrain.generate_pairs \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_<N>.pt \
  --output /workspace/dpo-pairs_raw.jsonl \
  --num-prompts 2500 --max-new-tokens 256 \
  > /root/dpo_generate_pairs.log 2>&1 &
disown
```

If it dies partway, resume with the exact same `--num-prompts` (prompt selection depends on
it) plus `--resume`:

```bash
nohup uv run --env-file .env python -m llmtrain.generate_pairs \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_<N>.pt \
  --output /workspace/dpo-pairs_raw.jsonl \
  --num-prompts 2500 --max-new-tokens 256 --resume \
  > /root/dpo_generate_pairs_resume.log 2>&1 &
disown
```

### 3b. Judge into preference pairs

Costs real money per API call (default: `together` / `meta-llama/Llama-3.3-70B-Instruct`,
double-evaluated per pair) — never re-run this without `--resume` against an `--output` path
that already has a partial/completed run on it, or you silently corrupt it (this has
happened for real; recovery is filtering to valid JSON lines).

```bash
nohup uv run --env-file .env python -m llmtrain.judge \
  --input /workspace/dpo-pairs_raw.jsonl \
  --output /workspace/dpo-pairs_dpo.jsonl \
  > /root/dpo_judge.log 2>&1 &
disown
```

Resume if interrupted:

```bash
nohup uv run --env-file .env python -m llmtrain.judge \
  --input /workspace/dpo-pairs_raw.jsonl \
  --output /workspace/dpo-pairs_dpo.jsonl --resume \
  > /root/dpo_judge_resume.log 2>&1 &
disown
```

If the default judge stops honoring structured outputs, try `--judge-provider cerebras`
(whatever strong instruct model is currently live there) or fall back to
`--judge-provider featherless-ai`.

### 3c. Train

```bash
nohup uv run --env-file .env python -m llmtrain.training.dpo \
  --checkpoint /workspace/sft-checkpoints-smoltalk/step_<N>.pt \
  --pairs /workspace/dpo-pairs_dpo.jsonl \
  --checkpoint-dir /workspace/dpo-checkpoints \
  --num-train-epochs 1 \
  > /root/dpo_train.log 2>&1 &
disown
```

Writes to its own `--checkpoint-dir` (never the SFT one — the DPO-exported step number
restarts from TRL's own `global_step`, not the SFT step count, so it would collide). No
`--resume` support today — real runs so far are short (~17 steps, minutes), so a crash just
means re-running this command; only worth adding resume plumbing if a run ever starts taking
a meaningful fraction of an hour (see `CLAUDE.md`).

### Evaluate the DPO checkpoint

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint /workspace/dpo-checkpoints/step_<N>.pt --chat --prompt "..."
```

## Pulling a checkpoint down for local evaluation

RunPod exposes network volumes over an S3-compatible API even when the pod isn't running —
`generate.py` reads `s3://` paths directly (requires `uv sync --extra s3` and an S3 API key
pair + endpoint/region in `.env`):

```dotenv
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL_S3=https://s3api-us-md-1.runpod.io
AWS_DEFAULT_REGION=us-md-1
```

```bash
uv run --env-file .env python -m llmtrain.generate \
  --checkpoint s3://<volume-id>/dpo-checkpoints/step_<N>.pt --chat --prompt "..."
```

List what's actually on the volume:

```bash
uv run --with boto3 --env-file .env python -c "
import boto3
s3 = boto3.client('s3')
for obj in s3.list_objects_v2(Bucket='<volume-id>').get('Contents', []):
    print(obj['Key'], obj['Size'])
"
```

## If a checkpoint save gets corrupted

`save_checkpoint()` writes atomically with retries, so this should be rare, but if a
`torch.load` on a checkpoint fails:

```bash
python3 -c "import zipfile; print(zipfile.ZipFile('/workspace/checkpoints/step_N.pt').testzip())"
# None = valid. A traceback/CRC mismatch = corrupt.
mv step_N.pt step_N.pt.corrupt   # move aside, don't delete
```

Then resume from the last good checkpoint as usual (see the `--resume` commands above).
