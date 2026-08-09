# Training guide

This is the full walkthrough for running this project end to end: a local smoke test on a
Mac, a real-scale smoke test on a rented A100, the actual `fineweb-edu` pretraining run, SFT
on top of it, and inference against a checkpoint with `generate.py`. See the repo root
`README.md` for a project overview; this doc assumes you've already read that and just want
to run things.

## Setup

This applies both on your Mac and on a rented pod — clone, install, configure. If `uv` isn't
already on `PATH` (true on a fresh RunPod container — official RunPod PyTorch images don't
ship it), install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Then:

```bash
git clone https://github.com/objones25/llm-from-scratch.git
cd llm-from-scratch
uv sync
```

`uv sync` reads `pyproject.toml`/`uv.lock` and creates a `.venv` with everything pinned there —
no separate `pip install` step. On a CUDA machine (the A100 pod, not your Mac), also install
the fused-cross-entropy dependency:

```bash
uv sync --extra cuda
```

Then create a `.env` file at the repo root (git-ignored — never commit it) with a Hugging
Face token and a Weights & Biases API key:

```dotenv
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

Get `HF_TOKEN` from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
(a read-scoped token is sufficient — this project only reads public datasets, it never
pushes anything to the Hub) and `WANDB_API_KEY` from
[wandb.ai/authorize](https://wandb.ai/authorize). Every command in this guide that needs
either variable is invoked as `uv run --env-file .env ...`; commands run with
`--wandb-mode disabled` (the local smoke test in Part 1) don't strictly need `WANDB_API_KEY`,
but `--env-file .env` is harmless to pass regardless, so the examples below use it
consistently.

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
| ---- | -------- |
| 10   | 8.0030   |
| 20   | 7.5120   |
| 30   | 7.2297   |

That decreasing trend is the real sanity signal to look for — not the absolute values, and
not (yet) the generated text quality; see below. Checkpoint pruning also behaved correctly:
with `--keep-last-n-checkpoints 2`, `step_10.pt` was deleted once `step_30.pt` was written,
leaving only `step_20.pt` and `step_30.pt` in `/tmp/smoke-test`.

**Important caveat about this run**: no `--d-model`/`--n-layers`/etc. override was passed, so
this ran the full default architecture at the time this transcript was captured — 100.7M
params (`d_model=768`, `n_layers=12`, `n_heads=12`, `n_kv_heads=4`), which is what produced
the exact `val_loss` numbers above. The default architecture is now the larger 253.8M-param
one (`d_model=1024`, `n_layers=20`, `n_heads=16`, `n_kv_heads=4`) used by the real pretraining
run — re-running this exact command today exercises that architecture instead, so expect
different absolute `val_loss` values and roughly 3GB checkpoint files (not the ~1GB this
original transcript produced) even at 30 steps. Either way it's slow on Mac CPU/MPS (tens of
seconds per optimizer step); the decreasing trend is still the signal to look for, not the
specific numbers. If you just want to confirm the pipeline runs end to end without waiting or
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
buffer and make `--resume` silently train zero steps (see Part 6). This is the dataset to use
to actually exercise `--resume` before trusting it on the real `fineweb_edu` run.

### One-time RunPod account setup

**Add an SSH key to your RunPod account before deploying anything.** RunPod injects your
public key into a pod's `~/.ssh/authorized_keys` automatically at launch, but only if the key
was already on your account first — adding it after the pod is already running means either
redeploying or pasting the key in manually. Generate one if you don't already have one you
want to reuse, then add the public half at
[runpod.io → Settings → SSH Public Keys](https://www.runpod.io/console/user/settings):

```bash
ssh-keygen -t ed25519 -C "runpod" -f ~/.ssh/runpod_ed25519
cat ~/.ssh/runpod_ed25519.pub   # paste this into the RunPod console
```

### Create a network volume

Network volumes are **tied to a specific data center** — the pod you deploy has to land in
that same data center to attach it, so create the volume first and pick the pod's data center
to match, not the other way around:

1. RunPod console → **Storage** → **New Network Volume**.
2. Pick a data center, give it a name, and size it generously above what `--keep-last-n-checkpoints`
   × ~3GB actually needs (see Part 3's cost-awareness note) — network volumes are billed for
   their provisioned size regardless of how much is actually used, and resizing later means
   picking a size you won't need to revisit mid-run.

### Deploy the pod

1. RunPod console → **Pods** → **Deploy**.
2. Filter to GPU pods in the **same data center as your network volume**, select an A100.
3. Choose a template with PyTorch/CUDA preinstalled (e.g. an official "RunPod PyTorch" image)
   — this project needs Python 3.12+ and a working CUDA toolchain, but installs everything
   else itself via `uv sync`, so the exact template matters less than getting the data center
   and GPU type right.
4. Attach the network volume you created above. Its mount path is commonly `/workspace`, but
   don't assume — confirm once connected (see "On the pod" below) before trusting any
   `--checkpoint-dir` value against it.
5. Choose **Spot** over On-Demand where available — checkpoint-on-network-volume plus this
   project's exact stream resume (`--resume`) absorbs interruption risk, and spot is
   meaningfully cheaper for a workload that can tolerate being preempted and resumed.
6. Deploy, wait for the pod to report Running, then connect:

   ```bash
   ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/runpod_ed25519
   ```

   (Find `<pod-ip>`/`<ssh-port>` on the pod's detail page, or via `runpodctl ssh info <pod-id>`
   if you have `runpodctl` installed locally.)

### On the pod

Once connected over SSH, follow this guide's [Setup](#setup) section above to clone the repo
and run `uv sync` — same steps as local, on the pod's own disk (not the network volume; the
network volume is for checkpoints via `--checkpoint-dir`, not the code checkout). Then,
pod-specific additions:

- Confirm the network volume's actual mount path before running anything that writes to it:
  `df -h | grep workspace`. Don't assume it's `/workspace` — use whatever that command actually
  shows, and pick a checkpoint subdirectory under it (e.g. `<mount>/checkpoints`) for
  `--checkpoint-dir` below. `train()` creates this directory itself
  (`checkpoint_dir.mkdir(parents=True, exist_ok=True)` in `training/train.py`), so there's no
  need to `mkdir` it by hand — just get the path right.
- `uv sync --extra cuda` (not needed on your Mac, required here) installs `liger-kernel` —
  `TrainConfig.use_fused_ce` defaults to `True`, so training will `ImportError` partway into a
  run (after tokenizer training and dataset streaming, not immediately) on a CUDA box that
  skips this step.
- Official RunPod PyTorch images don't ship `nano`. Write `.env` without an editor instead:
  ```bash
  cat > .env << 'EOF'
  HF_TOKEN=hf_your_actual_token_here
  WANDB_API_KEY=your_actual_wandb_key_here
  EOF
  ```
  (or `vi .env`, which is preinstalled, or `apt update && apt install -y nano` if you'd rather
  have the editor).
- `pip install wandb && wandb login` (or `uv run --env-file .env wandb login --verify` if
  `.env` is already set up) — confirm W&B auth works before starting a real run. If you're
  scripting pod setup (a startup script or custom template), bake this in so it survives fresh
  containers.
- Launch training with `nohup ... & disown` and disconnect — monitor progress via the W&B
  dashboard rather than holding the SSH session open. No inbound port exposure is needed; W&B
  is outbound-only. (`tmux` is the classic alternative but isn't installed on official RunPod
  PyTorch images — `apt install -y tmux` if you'd rather have it; `nohup`/`disown` need no
  install and are what every command in this guide uses.)

### Running the smoke test

The flag values below (`--max-steps`, `--checkpoint-interval`, `--eval-interval`) are a
reasonable starting point for a real-architecture ~15-minute run on an A100, chosen to land a
handful of checkpoints within that window — they are **not benchmarked against actual A100
throughput in this session**, so watch the first few step times in the log/W&B dashboard and
adjust `--max-steps` if 15 minutes runs long or short:

```bash
nohup uv run --env-file .env python -m llmtrain.training.train --dataset reformer_enwik8 \
  --max-steps 150 --checkpoint-interval 25 --eval-interval 50 --keep-last-n-checkpoints 3 \
  --checkpoint-dir /workspace/checkpoints --wandb-mode online \
  > /root/train.log 2>&1 &
disown
```

(`/workspace` assumes `df -h | grep workspace` showed that as the mount above — swap it if yours
differs.) Every training command in this guide is wrapped in `nohup ... & disown` from here on
— see Part 6's "dropped SSH connection" entry for why: without it, your SSH session dropping
(laptop sleep, network blip, closed terminal) kills the training process mid-write and corrupts
whatever checkpoint it was saving. `disown` detaches the background job from the shell so it
survives the session ending, not just a sleeping laptop. Tail `/root/train.log` to watch
progress, or just check the W&B dashboard instead of holding the terminal open.

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
nohup uv run --env-file .env python -m llmtrain.training.train --dataset reformer_enwik8 \
  --max-steps 150 --checkpoint-interval 25 --eval-interval 50 --keep-last-n-checkpoints 3 \
  --checkpoint-dir /workspace/checkpoints \
  --resume /workspace/checkpoints/step_50.pt --wandb-mode online \
  > /root/train_resume.log 2>&1 &
disown
```

Confirm in the logs/W&B that the step counter picks up at 50 rather than restarting at 0, and
that `loss`/`val_loss` continue their existing trend rather than jumping back up to their
step-0 values — that's the signal `--resume` actually restored model/optimizer state and
dataset stream position rather than just resuming the step counter cosmetically. Per
`CLAUDE.md`'s documented shuffle-buffer caveat (see Part 6), expect up to ~1000 rows to be
silently skipped across the resume boundary — bounded and practically invisible against
1.1M rows, unlike on `tiny_shakespeare`.

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
short: that run was already well past its own compute-optimal token count (~69
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
(~12.2GB)** for pretraining alone. Part 4's SFT runs afterward add their own checkpoints on
top of that while `step_10500.pt` still needs to stay on the volume (`--init-from-checkpoint`
reads it) — realistic peak usage across pretraining plus both SFT stages, if you don't clean
anything up in between, is closer to **~27.4GB** (12.2GB pretraining peak + ~3GB `no_robots`
checkpoints + ~12.2GB `smoltalk` peak). Your network volume needs to be resized to comfortably
clear that — **resize it to at least 30GB** (RunPod console → Storage → your volume → resize;
this generally requires stopping the pod first) before launching the command above, or delete
the older pretraining checkpoints (keeping only `step_10500.pt`) before starting Part 4 if
you'd rather stay closer to 20GB.

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

## Part 4 — SFT (`no_robots` sanity check, then `smoltalk`)

`--init-from-checkpoint` loads model weights only (no optimizer/step state) from a
pretraining checkpoint into a fresh SFT run — see `CLAUDE.md`'s architecture section for the
full `--init-from-checkpoint`/`--resume` distinction and footguns.

### 1. Sanity check on `no_robots`

Small, fast dataset — proves the SFT pipeline works before committing to the long `smoltalk`
run below. Points at the pretraining run's final checkpoint, `step_10500.pt`:

```bash
nohup uv run --env-file .env python -m llmtrain.training.train \
  --dataset no_robots \
  --init-from-checkpoint /workspace/checkpoints/step_10500.pt \
  --checkpoint-dir /workspace/sft-checkpoints \
  --max-seq-len 2048 \
  --lr 3e-5 --min-lr 3e-6 --warmup-steps 20 \
  --max-steps 150 \
  --wandb-project llm-training \
  > /root/train_sft_no_robots.log 2>&1 &
disown
```

(same `nohup ... & disown` pattern as pretraining — see Part 6's "dropped SSH connection"
entry. Tail `/root/train_sft_no_robots.log` or check the W&B dashboard.)

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
nohup uv run --env-file .env python -m llmtrain.training.train \
  --dataset smoltalk \
  --init-from-checkpoint /workspace/checkpoints/step_10500.pt \
  --checkpoint-dir /workspace/sft-checkpoints-smoltalk \
  --max-seq-len 2048 \
  --lr 3e-5 --min-lr 3e-6 --warmup-steps 300 \
  --max-steps 12000 \
  --wandb-project llm-training \
  > /root/train_sft_smoltalk.log 2>&1 &
disown
```

(same `nohup ... & disown` pattern as pretraining. Tail `/root/train_sft_smoltalk.log` or
check the W&B dashboard.)

- `smoltalk`'s `all` config has ~1.0M train rows; at the default effective batch size (256),
  one epoch is ~3900 steps. `--max-steps 12000` (~3 epochs) is a starting point based on
  typical SFT recipes, not a value tuned against this specific model — watch `val_loss` on
  the W&B dashboard and stop the run manually whenever it plateaus or you've seen enough,
  rather than treating 12000 as a number you must reach. Since this runs detached via
  `nohup ... & disown`, stop it with `pgrep -f 'dataset smoltalk'` to find its PID, then
  `kill <pid>` — not `Ctrl-C`, which only works on a foreground process. `--checkpoint-interval`
  (default 125) means there's always a recent checkpoint to grab whenever you decide to stop.
- `--warmup-steps 300` is a larger absolute warmup for a much longer run — about 2.5% of
  `smoltalk`'s 12,000 steps (vs. `no_robots`' 20/150 = 13.3%); shorter runs typically warm up
  over a larger fraction of total steps, so this isn't a literal 1:1 scale-up of the
  `no_robots` value, just proportionate to standard SFT practice.

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

## Part 5 — `generate.py` in depth

```text
python -m llmtrain.generate --checkpoint <path/to/step_N.pt> [--tokenizer-path PATH] \
    --prompt "..." [--max-new-tokens N] [--temperature F] [--repetition-penalty F] \
    [--top-k N] [--top-p F]
```

`--tokenizer-path` defaults to `tokenizer.json` next to the checkpoint (saved there by
`train.py`). Model architecture is reconstructed from the `model_config` persisted in the
checkpoint itself (falls back to `ModelConfig()` defaults for older checkpoints saved before
that field existed) — so architecture fields that change numerics without changing tensor
shapes (e.g. `rope_theta`) can't silently drift between training and generation.

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

## Part 6 — Troubleshooting and known limitations

**`--resume` silently drops examples across the resume boundary, catastrophically so on
`tiny_shakespeare`.** `datasets` (confirmed v5.0.1) doesn't preserve the shuffle buffer's
_contents_ across `state_dict()`/`load_state_dict()` — only enough to resume the underlying
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

**Checkpoint storage is a real, capped cost, not just tidiness.** See Part 3 — ~3.05GB per
checkpoint at the default architecture, capped at `--keep-last-n-checkpoints` (default 3) via
`prune_old_checkpoints()`.

**A checkpoint save can be interrupted mid-write and corrupt that checkpoint — now fixed at the
code level, but still recoverable manually on older checkpoints/checkouts.** Confirmed twice in
real runs, surfacing both times as `RuntimeError: basic_ios::clear: iostream error` out of
`torch.serialization.save` (`_open_zipfile_writer.__exit__` → `write_end_of_file`): once from a
local machine going to sleep and dropping the SSH session while `train.py` ran in the
foreground, and once under `nohup ... & disown` with no SSH session involved at all. The second
occurrence ruled out "dropped SSH" as the sole cause — `--checkpoint-dir` on the pod is a
network-mounted volume (MooseFS, confirmed via `df -h`), and the real root cause is a transient
write failure against that network mount, which a dropped SSH session can trigger but isn't
required for. `save_checkpoint()` (`src/llmtrain/training/checkpoint.py`) now writes to a
`step_N.pt.tmp` file and only `os.replace()`s it into place on success, retrying transient
`RuntimeError`/`OSError` failures up to 3 times with a 5s delay — so a blip like this can no
longer corrupt a checkpoint, and self-heals without losing the whole training process. If
you're on an older checkout without this fix, or the retries themselves are exhausted (the
network mount is down for longer than ~15s), recover manually:

The corrupt file can be a normal-looking size — checkpoints of the same model/optimizer shapes
are always similarly sized regardless of corruption, since only tensor _values_ differ, not
structure — so size alone doesn't confirm validity. To recover:

1. Identify the last checkpoint written successfully in the logs (`saved checkpoint at step N`)
   — the crash happens on the _next_ save attempt after that.

2. Confirm the suspect file is actually corrupt (official RunPod PyTorch images don't ship
   `unzip`, so use Python's `zipfile` module instead — a `torch.save` file is a zip archive):

   ```bash
   python3 -c "import zipfile; print(zipfile.ZipFile('/workspace/checkpoints/step_N.pt').testzip())"
   ```

   `None` means valid; a `BadZipFile`/CRC-mismatch result or traceback means corrupt.

3. Move the corrupt file aside rather than deleting it (`mv step_N.pt step_N.pt.corrupt`), in
   case you want to inspect it later.

4. Resume from the last good checkpoint, again under `nohup ... & disown` (see "On the pod"
   above) so a future disconnect can't kill the process again:

   ```bash
   nohup uv run --env-file .env python -m llmtrain.training.train --dataset fineweb_edu \
     --checkpoint-dir /workspace/checkpoints --wandb-mode online \
     --resume /workspace/checkpoints/step_<last-good>.pt \
     > /root/train_resume.log 2>&1 &
   disown
   ```

   `disown` detaches the background job from the current shell so it survives the SSH session
   itself ending, not just a sleeping laptop. Tail `/root/train_resume.log` to confirm step
   logging resumes, then rely on the W&B dashboard rather than holding the terminal open.
5. In W&B, the resumed run appears as a separate run with `step` continuing from where the
   crashed run left off (not restarting at 0) — group or overlay the two runs in the UI to view
   them as one continuous curve, since they share the same `step` axis.

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

```text
--dataset {tiny_shakespeare,reformer_enwik8,fineweb_edu,smoltalk,no_robots}
--shuffle-buffer-size, --max-seq-len, --tokenizer-vocab-size, --tokenizer-sample-size
--d-model, --n-layers, --n-heads, --n-kv-heads, --dropout, --rope-theta
--max-steps, --batch-size, --gradient-accumulation-steps, --grad-clip
--lr, --min-lr, --warmup-steps, --weight-decay, --beta1, --beta2, --seed
--checkpoint-dir, --checkpoint-interval, --keep-last-n-checkpoints, --eval-interval
--compile/--no-compile, --use-amp/--no-use-amp, --use-fused-ce/--no-use-fused-ce
--wandb-project, --wandb-mode {online,offline,disabled}, --log-file
--resume, --init-from-checkpoint, --tokenizer-path
```

`smoltalk`/`no_robots` (chat datasets, SFT) require either `--init-from-checkpoint` or
`--resume` — `main()` rejects them otherwise. `--tokenizer-path` requires
`--init-from-checkpoint`. See CLAUDE.md's architecture section for the `--resume` /
`--init-from-checkpoint` architecture-mismatch and `max_seq_len` footguns.

`generate.py`:

```text
--checkpoint (required), --tokenizer-path, --prompt (required)
--max-new-tokens, --temperature, --repetition-penalty, --top-k, --top-p
```
