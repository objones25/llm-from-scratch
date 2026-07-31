# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A toy LLM built from scratch and trained on `HuggingFaceFW/fineweb-edu`, using Hugging Face `tokenizers` and PyTorch. Full-scale training runs on a rented RunPod A100 GPU; smoke tests run locally on a Mac (MPS) first. The project is being scaffolded as a walking skeleton — see the full design at `docs/superpowers/specs/2026-07-31-project-scaffold-design.md` for anything not covered here.

A Hugging Face token is required for dataset/model access. Never commit it or write it into code — load it from the environment (e.g. `HF_TOKEN`) or the local HF CLI login.

## Datasets and their roles

| Dataset | Purpose |
|---|---|
| `karpathy/tiny_shakespeare` | Local smoke test (Mac/MPS) — fast, no GPU rental, validates the training loop end-to-end before spending money on a GPU. |
| `google/reformer-enwik8` | 15-minute A100 smoke test — validates loss/throughput numbers on real GPU hardware before committing to a full run. |
| `HuggingFaceFW/fineweb-edu` | Main pretraining corpus. Use `name="sample-100BT"`, `split="train"`, `streaming=True` — never download the full dataset. |
| `HuggingFaceTB/smoltalk` | SFT (supervised fine-tuning) after pretraining. |
| `HuggingFaceH4/no_robots` | Quick sanity checks (small, fast to iterate on). |

Workflow order: tiny_shakespeare (local) → reformer-enwik8 (A100, ~15 min) → fineweb-edu pretraining (A100) → smoltalk SFT, with no_robots available at any point for a fast sanity pass.

## Architecture

`src/llmtrain/` — one parameterized training entry point shared across every dataset above:

```
data/streaming.py    # load_dataset(..., streaming=True) wrapper, selected by DataConfig.dataset_name
data/tokenizer.py     # train/load a tokenizers.Tokenizer, independent of the model
model/transformer.py  # minimal model (architecture decisions are a separate spec)
training/config.py    # DataConfig/ModelConfig/TrainConfig dataclasses, no YAML layer
training/train.py      # train(data_cfg, model_cfg, train_cfg) — the only entry point
training/checkpoint.py # saves/loads model + optimizer + dataset iterator state as one unit
logging_config.py       # dictConfig: stdout + JSONL file handler
```

`python -m llmtrain.training.train --dataset <tiny_shakespeare|reformer_enwik8|fineweb_edu> --max-steps N` is the single entry point for local smoke tests, the A100 smoke test, and the real pretraining run — same code path every time, which is what makes a smoke test meaningful evidence the real run's path works.

## Device handling (MPS vs CUDA)

Verified against current PyTorch docs — don't assume from general knowledge, backend support here changes between versions:

- Select device with `torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")`.
- Always pass `pin_memory=True` to `DataLoader` — PyTorch itself forces it off on MPS (with a warning), so no branching is needed.
- Gate `torch.compile` behind `device.type == "cuda"`. The MPS inductor backend is an explicit prototype, limited to elementwise ops and excluded from fusion optimization — don't use it on Mac.
- `torch.autocast(device_type=device.type, ...)` works identically on `"cuda"` and `"mps"` — default dtype is `float16` on both.

## Dataset streaming & resume

`fineweb-edu` is loaded with `streaming=True` (never fully downloaded). For exact resume after an interruption, use `IterableDataset`'s built-in `state_dict()` / `load_state_dict()` — persist the state dict alongside the model checkpoint. Don't build custom skip/seed tracking; the library already does this.

## Logging & observability

Two systems, no overlap:

- **W&B** owns training metrics — loss, learning rate, tokens/sec, grad norm, eval metrics, GPU memory, checkpoints as artifacts.
- **JSONL structured logs** (`logging_config.py`, `dictConfig` + a custom `JSONFormatter`) own everything else — tokenizer progress, data pipeline warnings, checkpoint save/load events, exceptions, resolved config at CLI invocation. Always log via `logging.getLogger(__name__)`, never the root logger directly.

## RunPod workflow

- Rent **spot** instances — checkpoint-on-network-volume plus exact stream resume absorbs interruption risk, and spot is meaningfully cheaper.
- `TrainConfig.checkpoint_dir` points at a mounted **network volume** so checkpoints survive pod stop/restart independent of the pod's own disk.
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
```

## Testing strategy

Everything except the GPU training loop itself gets a real, fast, CPU-only unit test with tiny fake data (no GPU, no network, no cost) — data loading, tokenizer, model forward/backward shapes, checkpoint round-trip (including dataset iterator state), config parsing. The two GPU smoke tests (tiny_shakespeare locally, reformer-enwik8 on A100) are run manually and judged by loss/throughput in W&B, not automated as pass/fail tests.

## Development principles

- **Fail-fast TDD**: write a failing test before writing the implementation; keep feedback loops short, especially given GPU rental costs make late-discovered bugs expensive.
- **SOLID**: tokenizer, dataset loading, model, training loop, and evaluation are separable, substitutable concerns.
- **Karpathy principles for overengineering**: before adding abstraction, ask whether it earns its keep at the current scale of this toy project. Default to the simplest thing that works; prefer readable, hackable, single-purpose scripts over premature generalization. When in doubt about whether a component is overengineered, evaluate it against this standard rather than adding configurability "just in case."
