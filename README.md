# llm-training

A small language model built from scratch — custom transformer, tokenizer training, streaming
data pipeline, training loop, checkpointing, and KV-cache generation — pretrained on
[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), then
SFT'd on `HuggingFaceTB/smoltalk` and DPO-tuned on judged preference pairs, using Hugging Face
`tokenizers`/`transformers`/`trl` and PyTorch. This is a toy/educational project, not a
production framework: one model architecture, one training script per stage, no config-file
layer, no multi-GPU orchestration, no serving stack. Local development and smoke testing run on
a Mac (MPS/CPU); the real training runs happen on a rented RunPod A100 GPU.

## Architecture highlights

`TransformerLM` (`src/llmtrain/model/transformer.py`) is a decoder-only transformer with:

- **RoPE** (rotary position embeddings), computed fresh per forward call and offset-aware, so
  the same code path serves both full-sequence training and single-token cached decode.
- **RMSNorm** pre-normalization (`nn.RMSNorm`) around attention and MLP sublayers.
- **SwiGLU** MLP (`silu(w_gate(x)) * w_up(x)` fed through `w_down`), the LLaMA-style gated
  feed-forward, with the hidden dimension set to `2/3 * 4 * d_model`.
- **Grouped-query attention (GQA)**: separate `n_heads`/`n_kv_heads` (16/4 by default, a 4:1
  ratio) via `F.scaled_dot_product_attention(..., enable_gqa=True)`.
- **Weight-tied embeddings/head** — the output projection shares its weight tensor with the
  input token embedding (`self.head.weight = self.token_emb.weight`).
- **KV-cache-aware forward pass** (`model/cache.py`'s `KVCache`, threaded through via
  `position_offset`/`cache`/`layer_idx`) for `generate.py`. Note: SDPA's `is_causal` builds a
  top-left-aligned causal mask for non-square query/key lengths, which is wrong for cached
  decode, so masking is applied only when `seq_len > 1` (training / prefill); single-token
  cached decode needs no mask at all, since every cached key is a genuinely past position.
- **Fused cross-entropy** via [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
  (`LigerFusedLinearCrossEntropyLoss`) when `--use-fused-ce` is set — but it is only ever
  actually used on CUDA (`use_fused_ce and device.type == "cuda"`); on MPS/CPU training always
  falls back to a plain `F.cross_entropy` over the full logits.
- **Byte-level BPE tokenizer**, trained fresh at the start of every run from a small sample
  (`--tokenizer-sample-size`, default 10000 examples) drawn from the streaming dataset itself —
  there's no pre-shipped tokenizer artifact; `tokenizer.json` is saved next to the checkpoints.

At the default config (`d_model=1440`, `n_layers=20`, `n_heads=20`, `n_kv_heads=4`,
`vocab_size=32768`, `max_seq_len=2048`) the model is 478.6M parameters (431.4M
non-embedding).

## Requirements

- Python >= 3.12 (see `pyproject.toml`)
- [`uv`](https://docs.astral.sh/uv/) as the package/project manager
- A CUDA GPU only for the `cuda` extra (Liger Kernel fused cross-entropy) and for the real
  A100 pretraining run — everything else runs on MPS/CPU

## Installation

```bash
git clone https://github.com/objones25/llm-from-scratch.git
cd llm-from-scratch
uv sync
```

For the CUDA-only fused cross-entropy path (Liger Kernel), used on the RunPod A100 run, add
the `cuda` extra:

```bash
uv sync --extra cuda
```

## Environment setup

Training needs a Hugging Face token (to pull the gated/rate-limited datasets) and a
Weights & Biases API key (for metrics logging). Put both in a git-ignored `.env` file at the
repo root:

```dotenv
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

Never commit `.env` — it's already in `.gitignore`. Load it into any command with
`uv run --env-file .env <command>`.

## Quick example: local smoke test

This is a fast, tiny-scale smoke test on `tiny_shakespeare` — enough to confirm the pipeline
runs end-to-end on a Mac, not a meaningful training run. For the RunPod pod runbook (setup,
and the exact commands to run pretraining → SFT → DPO), see
[`docs/training-guide.md`](docs/training-guide.md).

Train for 30 steps:

```bash
uv run --env-file .env python -m llmtrain.training.train --dataset tiny_shakespeare \
  --max-steps 30 --gradient-accumulation-steps 2 --batch-size 4 --checkpoint-interval 10 \
  --eval-interval 10 --keep-last-n-checkpoints 2 --wandb-mode disabled \
  --checkpoint-dir /tmp/smoke-test --log-file /tmp/smoke-test.log
```

Generate from the resulting checkpoint:

```bash
uv run python -m llmtrain.generate --checkpoint /tmp/smoke-test/step_30.pt \
  --prompt "Once upon a time" --max-new-tokens 20
```

At this scale (30 steps, 472 training rows, default `temperature=1.0`) the output will be
largely incoherent — that's expected, not a bug; see Known limitations below.

## Datasets

| Dataset                        | Purpose                                                                                                                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Trelis/tiny-shakespeare`      | Local smoke test (Mac/MPS) — fast, no GPU rental. Its text column is `Text` (capital T); `data/streaming.py` renames it to `text`. Only 472 train rows — enough for a short smoke test, not for `--resume` (see Known limitations). |
| `reds0510/enwik8-processed`    | 15-minute A100 smoke test — 1.1M rows, real-scale enough for `--resume` to behave correctly.                                                                                                                                        |
| `HuggingFaceFW/fineweb-edu`    | Main pretraining corpus. `name="sample-100BT"`, `split="train"`, `streaming=True` — never downloaded in full.                                                                                                                       |
| `HuggingFaceTB/smoltalk`       | SFT (supervised fine-tuning) after pretraining.                                                                                                                                                                                     |
| `HuggingFaceH4/no_robots`      | Quick SFT sanity check (small, fast to iterate on) before committing to the long `smoltalk` run.                                                                                                                                    |
| `trl-lib/ultrafeedback-prompt` | Prompts `generate_pairs.py` samples completions from, for the DPO stage below.                                                                                                                                                      |

Workflow order: `tiny_shakespeare` (local) -> `reformer_enwik8` (A100, ~15 min) ->
`fineweb_edu` pretraining (A100) -> `no_robots`/`smoltalk` SFT -> DPO preference tuning.

## Repository structure

```text
src/llmtrain/
  data/streaming.py     DATASET_REGISTRY (dataset path/split/text-column config) + load_streaming_datasets
  data/tokenizer.py      Byte-level BPE training (train_tokenizer) and batch encode/pad (encode_batch)
  model/transformer.py   TransformerLM: RoPE, RMSNorm, SwiGLU, GQA, weight tying, KV-cache-aware forward
  model/cache.py          KVCache: per-layer (k, v) tensor cache, concatenated along the sequence dim
  model/hf_wrapper.py     TransformerLM wrapped as a transformers.PreTrainedModel, for TRL's DPOTrainer
  training/config.py     DataConfig/ModelConfig/TrainConfig/GenerationConfig dataclasses (single source of CLI defaults)
  training/train.py       select_device, loss functions, get_lr (warmup+cosine), train(), main() -- the training CLI
  training/checkpoint.py  save/load model + optimizer + dataset iterator state + model config, as one unit
  training/dpo.py         DPO training via TRL's DPOTrainer/DPOConfig -- the third DPO pipeline stage
  generate.py             KV-cache-backed text generation CLI (greedy or temperature/top-k/top-p sampling, --chat)
  generate_pairs.py       Samples completion pairs from an SFT checkpoint -- DPO pipeline stage 1
  judge.py                 LLM-as-judge: turns completion pairs into chosen/rejected preference pairs -- stage 2
  logging_config.py       dictConfig setup: stdout + JSONL file handler
```

Full CLI reference for `train.py`/`generate.py` below (or run any entry point with `--help`).
The DPO pipeline (`generate_pairs.py` → `judge.py` → `training/dpo.py`) is pod-only, unattended,
multi-stage — see [`docs/training-guide.md`](docs/training-guide.md) for the actual commands
rather than a flag reference here.

**`train.py`**:

```text
python -m llmtrain.training.train --dataset {tiny_shakespeare,reformer_enwik8,fineweb_edu,smoltalk,no_robots}
    [--shuffle-buffer-size N] [--max-seq-len N] [--tokenizer-vocab-size N] [--tokenizer-sample-size N]
    [--d-model N] [--n-layers N] [--n-heads N] [--n-kv-heads N] [--dropout F] [--rope-theta F]
    [--max-steps N] [--batch-size N] [--gradient-accumulation-steps N] [--grad-clip F]
    [--lr F] [--min-lr F] [--warmup-steps N] [--weight-decay F] [--beta1 F] [--beta2 F] [--seed N]
    [--checkpoint-dir DIR] [--checkpoint-interval N] [--keep-last-n-checkpoints N] [--eval-interval N]
    [--compile/--no-compile] [--use-amp/--no-use-amp] [--use-fused-ce/--no-use-fused-ce]
    [--wandb-project NAME] [--wandb-mode {online,offline,disabled}] [--log-file PATH]
    [--resume PATH] [--init-from-checkpoint PATH] [--tokenizer-path PATH]
```

`smoltalk`/`no_robots` are chat datasets (SFT) and require either `--init-from-checkpoint`
(starting an SFT run from a pretrained checkpoint) or `--resume`; `--tokenizer-path` requires
`--init-from-checkpoint`. See `CLAUDE.md`'s architecture section for the `--resume` /
`--init-from-checkpoint` architecture-mismatch footgun.

**`generate.py`**:

```text
python -m llmtrain.generate --checkpoint PATH --prompt "..." [--tokenizer-path PATH] [--chat]
    [--max-new-tokens N] [--temperature F] [--repetition-penalty F] [--top-k N] [--top-p F]
```

`--chat` wraps `--prompt` in the `<|user|>\n...\n<|assistant|>\n` shape every SFT/DPO training
example starts with — **required** for any chat-tuned checkpoint (`smoltalk`/`no_robots` SFT,
or DPO on top of either); omit it only for base/pretraining-only checkpoints. A raw prompt
against a chat-tuned checkpoint is out-of-distribution and produces garbled output that looks
like a model-quality problem but isn't — see `docs/dpo-run-results.md` §4 for a worked example.

Every `train.py` flag's default reads from the corresponding `DataConfig`/`ModelConfig`/
`TrainConfig` field (`training/config.py`); every `generate.py` sampling flag's default reads
from `GenerationConfig`. Nothing is duplicated as a literal in the argparse setup.

## Development commands

```bash
uv run pytest                              # run all tests
uv run pytest tests/test_foo.py::test_bar  # run a single test
uv run ruff check .                        # lint
uv run ruff format .                       # format
uv run mypy src/                           # type check
uv run --env-file .env wandb login --verify  # confirm W&B auth before a real run
```

Everything except the GPU training loop itself has a real, fast, CPU-only unit test with tiny
fake data (data loading, tokenizer, model forward/backward shapes, checkpoint round-trip,
config parsing). `train()`/`main()` orchestration has no automated test by design — it's
validated by the manual smoke test described above.

## Known limitations

- **`--resume` silently does nothing useful on `tiny_shakespeare`.** Its 472 rows are smaller
  than the default shuffle buffer (1000), and `datasets` discards the shuffle buffer's
  contents on `load_state_dict`, so the resumed stream comes up empty with no error raised.
  `--resume` works correctly on real-scale datasets (`reformer_enwik8`, `fineweb_edu`).
- **Checkpoints are large even at smoke-test scale.** At the default 478.6M-parameter
  architecture, each `step_N.pt` is roughly 5.74GB (model + optimizer state), regardless of
  how few training steps produced it. `--keep-last-n-checkpoints` prunes old ones, but disk
  usage during a run should be planned for accordingly.
- **Short runs produce garbled output.** A few dozen steps on a few hundred training rows,
  sampled at the default `temperature=1.0`, is not enough signal for coherent generation —
  this is expected behavior for a smoke test, not a bug in `generate.py`.
- **Fused cross-entropy (`--use-fused-ce`) is CUDA-only.** The flag itself is accepted
  everywhere, but the fused path is only actually used when `device.type == "cuda"`; on
  MPS/CPU it silently falls back to the standard loss.
- **`generate.py --chat` is required for chat-tuned checkpoints, and it's easy to forget.**
  There's no automatic detection of whether a checkpoint was chat-tuned — omitting `--chat`
  against an SFT/DPO checkpoint silently produces garbled, off-topic output that looks like a
  model-quality problem rather than a missing flag. See the CLI reference above.
- **DPO training has no `--resume`.** A crash mid-run loses all training progress (not the
  judged preference data, which is written independently). Deliberate, not an oversight — real
  runs so far take ~17 optimizer steps (minutes), so this isn't worth the complexity yet; see
  `CLAUDE.md`'s DPO pipeline section if that changes.

See `CLAUDE.md` for the full technical reference (device handling, RunPod workflow, DPO
pipeline, logging strategy, and more), and [`docs/training-guide.md`](docs/training-guide.md)
for the pod runbook — setup and the exact commands to run pretraining → SFT → DPO.
