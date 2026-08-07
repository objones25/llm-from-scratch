# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A toy LLM built from scratch and trained on `HuggingFaceFW/fineweb-edu`, using Hugging Face `tokenizers` and PyTorch. Full-scale training runs on a rented RunPod A100 GPU; smoke tests run locally on a Mac (MPS) first. The walking-skeleton pipeline is implemented and its local smoke test passed end-to-end — see `docs/superpowers/specs/2026-07-31-project-scaffold-design.md` for the original design. (The smoke-test walkthrough doc was removed and hasn't been replaced yet — see Testing strategy below.)

A Hugging Face token and a W&B API key are required (`HF_TOKEN`, `WANDB_API_KEY` in a git-ignored `.env`, loaded via `uv run --env-file .env ...`). Never commit either.

## Datasets and their roles

| Dataset                     | Purpose                                                                                                                                                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Trelis/tiny-shakespeare`   | Local smoke test (Mac/MPS) — fast, no GPU rental. Its text column is `Text` (capital T); `data/streaming.py` renames it to `text` via `DatasetSpec.text_column`. Only 472 train rows — enough for a short smoke test, not for `--resume` (see below). |
| `reds0510/enwik8-processed` | 15-minute A100 smoke test — 1.1M rows, real-scale enough for `--resume` to behave correctly.                                                                                                                                                          |
| `HuggingFaceFW/fineweb-edu` | Main pretraining corpus. `name="sample-100BT"`, `split="train"`, `streaming=True` — never download the full dataset.                                                                                                                                  |
| `HuggingFaceTB/smoltalk`    | SFT (supervised fine-tuning) after pretraining.                                                                                                                                                                                                       |
| `HuggingFaceH4/no_robots`   | Quick sanity checks (small, fast to iterate on).                                                                                                                                                                                                      |

`karpathy/tiny_shakespeare` and `google/reformer-enwik8` (the original picks) are dead on the Hub — script-only and deleted, respectively — hence the replacements above. Workflow order: tiny_shakespeare (local) → reformer-enwik8-processed (A100, ~15 min) → fineweb-edu pretraining (A100) → smoltalk SFT.

Validation strategy is per dataset: `tiny_shakespeare` uses its native `test` split; `reformer_enwik8`/`fineweb_edu` carve a 1000-example holdout from the shuffled train stream instead (no native val split). See `data/streaming.py`'s `load_streaming_datasets` in Architecture below.

## Architecture

`src/llmtrain/` — one parameterized training entry point shared across every dataset above:

```
data/streaming.py    # DATASET_REGISTRY (DatasetSpec incl. text_column rename) + load_streaming_datasets,
                       # returning (train: IterableDataset, val: list[dict]) — val is always materialized
                       # so it never shares mutable streaming state with train's IterableDataset
data/tokenizer.py     # train_tokenizer / encode_batch, independent of the model
model/transformer.py  # TransformerLM: RoPE, RMSNorm, SwiGLU MLP, weight-tied embeddings/head,
                       # GQA, hand-rolled causal attention via F.scaled_dot_product_attention with
                       # is_causal=(seq_len > 1) — SDPA's non-square causal bias is top-left-aligned,
                       # not cached-decode-safe, so masking is skipped for single-token cached decode
                       # instead — and a KV-cache-aware forward (position_offset/cache/layer_idx threaded through)
model/cache.py         # KVCache: per-layer (k, v) tensor cache, update() concatenates along seq dim
training/config.py    # DataConfig/ModelConfig/TrainConfig/GenerationConfig dataclasses, no YAML layer
training/train.py      # select_device, next_token_loss, make_collate_fn, train(), main() — the training entry point
training/checkpoint.py # saves/loads model + optimizer + dataset iterator state + model architecture config as one unit
generate.py             # KV-cache-backed text generation from a checkpoint; main() is a second CLI entry point
logging_config.py       # dictConfig: stdout + JSONL file handler
```

```
python -m llmtrain.training.train --dataset <tiny_shakespeare|reformer_enwik8|fineweb_edu> \
    [--max-steps N] [--batch-size N] [--lr F] [--checkpoint-dir DIR] [--resume PATH]
```

Same code path for local smoke tests, the A100 smoke test, and the real pretraining run — `--checkpoint-dir` is the only thing that needs to change for RunPod (point it at the mounted network volume). Every `DataConfig`/`ModelConfig`/`TrainConfig` field is exposed as a CLI flag (`--d-model`, `--n-layers`, `--n-kv-heads`, `--dropout`, `--rope-theta`, `--min-lr`, `--warmup-steps`, `--weight-decay`, `--beta1`/`--beta2`, `--gradient-accumulation-steps`, `--grad-clip`, `--seed`, `--eval-interval`, `--compile`/`--no-compile`, `--use-amp`/`--no-use-amp`, `--wandb-project`, `--wandb-mode`, etc. — run `--help` for the full list); each flag's default reads from the corresponding dataclass field (e.g. `default=TrainConfig.checkpoint_interval`) rather than a duplicated literal, so the dataclasses are the single source of truth for defaults.

`train.py`'s `get_lr(step, train_cfg)` is a linear-warmup-then-cosine-decay schedule (nanoGPT-style): ramps from 0 to `lr` over `warmup_steps`, cosine-decays to `min_lr` by `max_steps`, then holds at `min_lr`. It's a pure function of `step`, so `--resume` needs no separate scheduler state — the restored step counter alone determines the LR. The optimizer's `param_groups` are updated every step (before `optimizer.step()`), and the resulting LR is logged to W&B alongside loss. **`step` means an optimizer step, not a micro-batch forward/backward** — `train()` accumulates gradients over `TrainConfig.gradient_accumulation_steps` (default 8) micro-batches before each `optimizer.step()`/`step += 1`, so `--max-steps`, `checkpoint_interval`, and W&B's `step` axis all count optimizer steps; `--resume` always lands on a clean accumulation-window boundary, so no accumulation state needs to be persisted. Gradients are clipped (`torch.nn.utils.clip_grad_norm_`, `TrainConfig.grad_clip`, default 1.0) once per window, right before `optimizer.step()`; the pre-clip norm is logged to W&B as `grad_norm`. AdamW uses two param groups — `weight_decay` (default 0.1) applies only to parameters with `dim() >= 2`, excluding `nn.RMSNorm` gains (the only remaining 1-D params, since every `nn.Linear` layer is `bias=False`); `beta2` defaults to `0.95` (not PyTorch's `0.999`), matching GPT-3/LLaMA-style pretraining practice. `TransformerLM`'s weights use LLaMA-style init (`_init_weights`: `N(0, 0.02²)` for `nn.Linear`/`nn.Embedding`, with `attn.out_proj`/`mlp.w_down` additionally scaled by `1/√(2·n_layers)` since they write directly into the residual stream).

Three more cheap, safe additions live in `train()`: `torch.set_float32_matmul_precision("high")` gated behind `device.type == "cuda"` (TF32 matmul speedup on Ampere+, no-op on MPS/CPU); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set via `os.environ.setdefault` before any CUDA allocation (reduces fragmentation-driven OOMs on long runs); and `drop_last=True` on the `DataLoader` (keeps every batch shape constant so `torch.compile` never recompiles mid-run on a ragged final batch).

Deferred (discussed, not implemented — revisit only if the problem they solve actually shows up): sequence packing (concat + chunk instead of pad — real throughput win, but needs reworking the tokenizer/collate pipeline, not a config tweak), DataLoader `num_workers`/`persistent_workers`/`prefetch_factor` tuning, fused cross-entropy, and gradient checkpointing — the last two solve memory pressure this config doesn't currently have.

A held-out validation loop is implemented: `evaluate()` runs every `TrainConfig.eval_interval` steps (default 500), logging `val_loss` to W&B alongside train `loss`. `load_streaming_datasets` (`data/streaming.py`) returns `(train, val)`, with validation strategy per dataset — `tiny_shakespeare` uses its native `test` split (`DatasetSpec.val_split`); `reformer_enwik8`/`fineweb_edu` carve a 1000-example holdout from the shuffled train stream (`DatasetSpec.val_holdout_examples`). Either way `val` comes back fully materialized as a `list[dict]`, never a lazy `IterableDataset` — this is deliberate: an early version left the carve path's val/train split lazy (`shuffled.take()`/`shuffled.skip()` sharing one underlying stream object), and iterating val mid-training silently corrupted the train stream's `state_dict()` tracking, making `--resume` rewind and silently retrain on already-seen data.

A second CLI entry point, `generate.py`, runs inference from a trained checkpoint (greedy or temperature-sampled decoding, KV-cache-backed, with repetition penalty and top-k/top-p sampling):

```
python -m llmtrain.generate --checkpoint <path/to/step_N.pt> [--tokenizer-path PATH] \
    --prompt "..." [--max-new-tokens N] [--temperature F] [--repetition-penalty F] [--top-k N] [--top-p F]
```

`--tokenizer-path` defaults to `tokenizer.json` next to the checkpoint (saved alongside checkpoints by `train.py`). Sampling defaults live in `GenerationConfig` (`training/config.py`), shared as the single source of truth between the CLI flag defaults and the `generate()`/`generate_token_ids()` signatures (both take a `GenerationConfig` instead of five separate parameters). Model architecture is reconstructed from the `model_config` persisted in the checkpoint (falls back to `ModelConfig()` defaults for older checkpoints saved before that field existed).

## Device handling (MPS vs CUDA)

Verified against current PyTorch docs — don't assume from general knowledge, backend support here changes between versions:

- Select device with `torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")`.
- Always pass `pin_memory=True` to `DataLoader` — PyTorch itself forces it off on MPS (with a warning), so no branching is needed.
- Gate `torch.compile` behind `device.type == "cuda"`. The MPS inductor backend is an explicit prototype, limited to elementwise ops and excluded from fusion optimization — don't use it on Mac.
- `torch.autocast(device_type=device.type, dtype=..., ...)`: `bfloat16` on CUDA (A100 supports it natively, no `GradScaler` needed); default dtype (`float16`) on MPS/CPU.
- `model` in `train()` is explicitly annotated `torch.nn.Module` — needed because `torch.compile`'s return type is a broad callable in the stubs, and without the annotation type checkers widen every later use of `model` to a union.

## Dataset streaming & resume

`fineweb-edu` is loaded with `streaming=True` (never fully downloaded). Exact resume uses `IterableDataset`'s built-in `state_dict()` / `load_state_dict()`, wired through `--resume <checkpoint path>` in `train()`: it restores model/optimizer state, the dataset's stream position, and continues the step counter (not restart at 0). **Known limitation:** `datasets` (confirmed v5.0.1) doesn't preserve the shuffle buffer's *contents* across `state_dict()`/`load_state_dict()` — only enough to resume the underlying stream position. On `load_state_dict`, it refills the buffer by reading `buffer_size` (default 1000) new elements from the stream before yielding again, and those refill elements are never yielded themselves, so every `--resume` permanently drops up to `buffer_size` examples. This is a property of `.shuffle()` itself, not specific to any one dataset in `DATASET_REGISTRY` — confirmed by isolating `.skip()` (round-trips exactly on its own) from `.shuffle()` (loses `buffer_size` examples on its own) in `tests/test_streaming.py::test_shuffled_skip_dataset_resumes_correctly_via_state_dict` (marked `xfail`, not fixed — see that test for the reproduction). It's catastrophic only on `tiny_shakespeare`: its 472 rows are smaller than the default shuffle buffer (1000), so the resumed stream comes up completely empty and the run silently trains zero steps, with no error raised. On `reformer_enwik8`/`fineweb-edu` the same mechanism only drops ~1000 rows out of millions/billions per resume — a bounded, practically invisible loss, not a stream failure. Not worth fixing for a dataset (`tiny_shakespeare`) that only exists for a 3-second smoke test, and not worth fixing generally given how small the loss is at real scale; be aware of it if `--resume` is ever tested against `tiny_shakespeare` specifically, where it is total rather than bounded.

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

Everything except the GPU training loop itself gets a real, fast, CPU-only unit test with tiny fake data (no GPU, no network, no cost) — data loading, tokenizer, model forward/backward shapes, checkpoint round-trip (including dataset iterator state), config parsing. `train()`/`main()` orchestration has no automated test by design — it was validated by an end-to-end manual smoke test (50 steps, tiny_shakespeare, decreasing loss, valid checkpoint, valid JSONL log) before the pretraining-loop-hardening changes (gradient accumulation, clipping, AdamW retune, weight init) landed. The walkthrough doc for that smoke test (`docs/smoke-test.md`) was deleted and hasn't been replaced yet — re-run a manual smoke test by hand (e.g. `uv run python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps 4 --gradient-accumulation-steps 2 --batch-size 2 --checkpoint-interval 2 --wandb-mode disabled`) until a new doc exists.

## Development principles

- **Fail-fast TDD**: write a failing test before writing the implementation; keep feedback loops short, especially given GPU rental costs make late-discovered bugs expensive.
- **SOLID**: tokenizer, dataset loading, model, training loop, and evaluation are separable, substitutable concerns.
- **Karpathy principles for overengineering**: before adding abstraction, ask whether it earns its keep at the current scale of this toy project. Default to the simplest thing that works; prefer readable, hackable, single-purpose scripts over premature generalization. When in doubt about whether a component is overengineered, evaluate it against this standard rather than adding configurability "just in case."
