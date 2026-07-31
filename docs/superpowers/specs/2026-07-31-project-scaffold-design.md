# Toy LLM Training Project — Scaffolding Design

Date: 2026-07-31

## Scope

This is a walking-skeleton scaffold: enough of the data → tokenizer → model → training loop → logging path to run the `tiny_shakespeare` smoke test locally end-to-end, proving the pipeline before spending money on GPU time.

**Explicitly deferred to later specs:** SFT module, PPO/DPO module, inference/API layer, and model architecture specifics (layer count, attention variant, etc.).

## Package layout

```
llm-training/
  src/llmtrain/
    __init__.py
    data/
      streaming.py       # load_dataset(..., streaming=True) wrapper per dataset
      tokenizer.py        # train/load a HF tokenizers.Tokenizer
    model/
      transformer.py      # minimal model, just enough to train (arch TBD separately)
    training/
      config.py           # DataConfig, ModelConfig, TrainConfig dataclasses
      train.py             # single parameterized train() entry point
      checkpoint.py        # save/load model + optimizer + dataset iterator state
    logging_config.py       # dictConfig setup + JSONFormatter
  tests/
    test_streaming.py
    test_tokenizer.py
    test_model.py
    test_checkpoint.py
    test_config.py
  pyproject.toml            # uv-managed
  CLAUDE.md
```

## Data pipeline

- `data/streaming.py` wraps `load_dataset(..., streaming=True)` per dataset, selected via `DataConfig.dataset_name` (`"tiny_shakespeare"`, `"reformer_enwik8"`, `"fineweb_edu"`). For `fineweb_edu`: `HuggingFaceFW/fineweb-edu`, `name="sample-100BT"`, `split="train"`.
- Shuffling: `.shuffle(seed=..., buffer_size=...)`.
- **Resume**: use `IterableDataset.state_dict()` / `.load_state_dict()` (built into `datasets`) for exact resume of stream position after an interruption. Persist the state dict alongside the model checkpoint; no custom skip/seed tracking needed.
- `data/tokenizer.py` trains/loads a `tokenizers.Tokenizer` from a dataset iterator, kept independent of the model so it's testable on a handful of in-memory strings with no network access.

## Config

Plain dataclasses in `training/config.py` — `DataConfig`, `ModelConfig` (minimal placeholder pending the architecture discussion), `TrainConfig` — with defaults, overridable via CLI flags, directly constructable in tests (no YAML layer).

## Device handling (MPS vs CUDA)

Verified against current PyTorch docs, not assumed from training data:

- **Device selection**: `device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")`.
- **`pin_memory`**: always pass `pin_memory=True` in the `DataLoader`. PyTorch forces it off on MPS itself (with a warning) — no branching needed in our code.
- **`torch.compile`**: gate behind `device.type == "cuda"` in `TrainConfig`. The MPS inductor backend is an explicit prototype ("not a feature-complete compiler backend"), limited to elementwise ops and excluded from fusion optimization — not worth attempting there.
- **`torch.autocast`**: pass `device_type=device.type` (`"cuda"` or `"mps"`); default autocast dtype is `float16` on both, so the call shape is identical across backends.

## Training entry point

One `train(data_cfg, model_cfg, train_cfg)` function invoked via `python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps 50` locally, and with `--dataset reformer_enwik8` / `--dataset fineweb_edu` for A100 runs — same code path every time, which is what makes the smoke tests meaningful evidence the real run's path works.

## Checkpointing

`training/checkpoint.py` saves/loads model + optimizer state + the dataset iterator's `state_dict()` as one unit, tested locally with a tiny model and a temp dir. On RunPod, `TrainConfig.checkpoint_dir` points at a mounted network volume — no code difference, only a config value changes between local and RunPod runs.

## Logging & observability

- **JSONL structured logging** (`logging_config.py`): `dictConfig` with a stdout handler (simple format, INFO+) and a rotating file handler using the provided `JSONFormatter` (DEBUG+, JSONL). Covers application-level events: tokenizer training progress, data pipeline warnings, checkpoint save/load events, exceptions, resolved config at CLI invocation. Every module uses `logging.getLogger(__name__)`.
- **W&B** (training metrics): `wandb.init()`/`wandb.log()` for loss, learning rate, tokens/sec, grad norm, eval metrics, GPU memory; checkpoints optionally logged as artifacts.
- These two systems don't overlap: JSONL never carries metrics, W&B never carries error/event logs.

## RunPod workflow

- Rent a **spot** A100 (cheaper; checkpoint-on-network-volume plus exact stream resume absorbs interruption risk).
- `checkpoint_dir` points at a mounted **network volume** (`/workspace/checkpoints`), surviving pod stop/restart independent of the pod's own disk.
- `pip install wandb && wandb login` once; bake into a startup script/Dockerfile if using a custom template so it survives fresh containers.
- Launch training inside **tmux** (or `nohup ... &`), then disconnect — monitor via the W&B dashboard, reattach to tmux only if a live terminal is needed.
- No inbound port exposure needed — W&B is outbound-only HTTPS from the pod.

## Testing strategy

Everything except the GPU training loop itself gets a real CPU-only unit test with tiny fake data — no GPU, no network, no cost:

- `test_streaming.py` — fake/tiny iterable stands in for the real streaming dataset; verifies shuffle/state_dict-resume logic without hitting the Hub.
- `test_tokenizer.py` — trains on a handful of in-memory sentences.
- `test_model.py` — forward/backward shape checks on a minimal config.
- `test_checkpoint.py` — save/load round-trip in a temp dir, including dataset iterator state.
- `test_config.py` — dataclass defaults, CLI override parsing.

The two GPU smoke tests (`tiny_shakespeare` locally, `reformer_enwik8` on A100) are run manually and validated by eyeballing loss/throughput in W&B — not automated pass/fail tests. Fail-fast TDD applies to every module above the GPU boundary: write the failing test first, then the minimal implementation.

## Tooling

uv-managed project. Core deps: `torch`, `tokenizers`, `datasets`, `wandb`.

```toml
[dependency-groups]
dev = ["pytest", "ruff", "mypy"]
```

Commands: `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`, `uv run mypy src/`.
