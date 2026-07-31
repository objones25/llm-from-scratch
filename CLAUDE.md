# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A toy LLM built from scratch and trained on `HuggingFaceFW/fineweb-edu`, using Hugging Face `tokenizers` and PyTorch. Full-scale training runs on a rented RunPod A100 GPU; smoke tests run locally on a Mac (MPS) first. The walking-skeleton pipeline is implemented and its local smoke test has passed end-to-end — see `docs/superpowers/specs/2026-07-31-project-scaffold-design.md` for the original design and `docs/smoke-test.md` for how to re-run verification.

A Hugging Face token and a W&B API key are required (`HF_TOKEN`, `WANDB_API_KEY` in a git-ignored `.env`, loaded via `uv run --env-file .env ...`). Never commit either.

## Datasets and their roles

| Dataset | Purpose |
|---|---|
| `Trelis/tiny-shakespeare` | Local smoke test (Mac/MPS) — fast, no GPU rental. Its text column is `Text` (capital T); `data/streaming.py` renames it to `text` via `DatasetSpec.text_column`. Only 472 train rows — enough for a short smoke test, not for `--resume` (see below). |
| `reds0510/enwik8-processed` | 15-minute A100 smoke test — 1.1M rows, real-scale enough for `--resume` to behave correctly. |
| `HuggingFaceFW/fineweb-edu` | Main pretraining corpus. `name="sample-100BT"`, `split="train"`, `streaming=True` — never download the full dataset. |
| `HuggingFaceTB/smoltalk` | SFT (supervised fine-tuning) after pretraining. |
| `HuggingFaceH4/no_robots` | Quick sanity checks (small, fast to iterate on). |

`karpathy/tiny_shakespeare` and `google/reformer-enwik8` (the original picks) are dead on the Hub — script-only and deleted, respectively — hence the replacements above. Workflow order: tiny_shakespeare (local) → reformer-enwik8-processed (A100, ~15 min) → fineweb-edu pretraining (A100) → smoltalk SFT.

## Architecture

`src/llmtrain/` — one parameterized training entry point shared across every dataset above:

```
data/streaming.py    # DATASET_REGISTRY (DatasetSpec incl. text_column rename) + load_streaming_dataset
data/tokenizer.py     # train_tokenizer / encode_batch, independent of the model
model/transformer.py  # MinimalTransformerLM: hand-rolled causal attention via F.scaled_dot_product_attention(is_causal=True)
training/config.py    # DataConfig/ModelConfig/TrainConfig dataclasses, no YAML layer
training/train.py      # select_device, next_token_loss, make_collate_fn, train(), main() — the only entry point
training/checkpoint.py # saves/loads model + optimizer + dataset iterator state as one unit
logging_config.py       # dictConfig: stdout + JSONL file handler
```

```
python -m llmtrain.training.train --dataset <tiny_shakespeare|reformer_enwik8|fineweb_edu> \
    [--max-steps N] [--batch-size N] [--lr F] [--checkpoint-dir DIR] [--resume PATH]
```
Same code path for local smoke tests, the A100 smoke test, and the real pretraining run — `--checkpoint-dir` is the only thing that needs to change for RunPod (point it at the mounted network volume).

## Device handling (MPS vs CUDA)

Verified against current PyTorch docs — don't assume from general knowledge, backend support here changes between versions:

- Select device with `torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")`.
- Always pass `pin_memory=True` to `DataLoader` — PyTorch itself forces it off on MPS (with a warning), so no branching is needed.
- Gate `torch.compile` behind `device.type == "cuda"`. The MPS inductor backend is an explicit prototype, limited to elementwise ops and excluded from fusion optimization — don't use it on Mac.
- `torch.autocast(device_type=device.type, dtype=..., ...)`: `bfloat16` on CUDA (A100 supports it natively, no `GradScaler` needed); default dtype (`float16`) on MPS/CPU.
- `model` in `train()` is explicitly annotated `torch.nn.Module` — needed because `torch.compile`'s return type is a broad callable in the stubs, and without the annotation type checkers widen every later use of `model` to a union.

## Dataset streaming & resume

`fineweb-edu` is loaded with `streaming=True` (never fully downloaded). Exact resume uses `IterableDataset`'s built-in `state_dict()` / `load_state_dict()`, wired through `--resume <checkpoint path>` in `train()`: it restores model/optimizer state, the dataset's stream position, and continues the step counter (not restart at 0). **Known limitation:** resuming works correctly on real-scale datasets (`reds0510/enwik8-processed`, `fineweb-edu`) but silently trains zero steps on `tiny_shakespeare` specifically — its 472 rows are smaller than the default shuffle buffer (1000), and `datasets` discards the shuffle buffer's contents on `load_state_dict`, so the resumed stream comes up empty. No error is raised. Not worth fixing for a dataset that only exists for a 3-second smoke test; be aware of it if `--resume` is ever tested against `tiny_shakespeare`.

## Logging & observability

Two systems, no overlap:

- **W&B** owns training metrics — loss, learning rate, tokens/sec, grad norm, eval metrics, GPU memory, checkpoints as artifacts.
- **JSONL structured logs** (`logging_config.py`, `dictConfig` + a custom `JSONFormatter`) own everything else — pipeline events, checkpoint save/load events, exceptions. Always log via `logging.getLogger(__name__)`, never the root logger directly. Note: the root logger is currently DEBUG with `disable_existing_loggers: False`, so `app.log` also fills with third-party DEBUG noise (`httpx`, `datasets`, etc.) — confirmed in a real run. Harmless (still valid JSONL) but noisy; scoping `llmtrain`'s logger to DEBUG and root to WARNING would clean it up if it becomes a problem.

## RunPod workflow

- Rent **spot** instances — checkpoint-on-network-volume plus exact stream resume absorbs interruption risk, and spot is meaningfully cheaper.
- `--checkpoint-dir` points at a mounted **network volume** so checkpoints survive pod stop/restart independent of the pod's own disk.
- `pip install wandb && wandb login` once per pod; bake into a startup script/Dockerfile for custom templates so it survives fresh containers.
- Launch training inside **tmux** (or `nohup ... &`) and disconnect — monitor via the W&B dashboard, don't hold the SSH session open. No inbound port exposure is needed; W&B is outbound-only.

## Commands

uv-managed project (`pyproject.toml`, `[dependency-groups] dev = ["pytest", "ruff", "mypy"]`):

```
uv run pytest                 # run all tests
uv run pytest tests/test_foo.py::test_bar   # run a single test
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src/              # type check
uv run --env-file .env wandb login --verify   # confirm W&B auth before a real run
```

## Testing strategy

Everything except the GPU training loop itself gets a real, fast, CPU-only unit test with tiny fake data (no GPU, no network, no cost) — data loading, tokenizer, model forward/backward shapes, checkpoint round-trip (including dataset iterator state), config parsing. `train()`/`main()` orchestration has no automated test by design — it's validated by the manual smoke test in `docs/smoke-test.md`, which has passed end-to-end (50 steps, tiny_shakespeare, decreasing loss, valid checkpoint, valid JSONL log).

## Development principles

- **Fail-fast TDD**: write a failing test before writing the implementation; keep feedback loops short, especially given GPU rental costs make late-discovered bugs expensive.
- **SOLID**: tokenizer, dataset loading, model, training loop, and evaluation are separable, substitutable concerns.
- **Karpathy principles for overengineering**: before adding abstraction, ask whether it earns its keep at the current scale of this toy project. Default to the simplest thing that works; prefer readable, hackable, single-purpose scripts over premature generalization. When in doubt about whether a component is overengineered, evaluate it against this standard rather than adding configurability "just in case."
