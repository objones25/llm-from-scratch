# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A toy LLM built from scratch and trained on `HuggingFaceFW/fineweb-edu`, using Hugging Face `tokenizers` and PyTorch. Full-scale training runs on a rented RunPod A100 GPU; smoke tests run locally on a Mac (MPS) first. The walking-skeleton pipeline is implemented and its local smoke test passed end-to-end — see `docs/superpowers/specs/2026-07-31-project-scaffold-design.md` for the original design, `README.md` for a project overview, and `docs/training-guide.md` for the full walkthrough (local smoke test, A100 smoke test, the real pretraining run, and troubleshooting).

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
data/chat.py           # chat-formatted encoding for SFT datasets (smoltalk, no_robots):
                       # wraps turns in role tags, masks non-assistant turns with IGNORE_INDEX,
                       # and splices a [PAD] token after each assistant turn as a stop signal
                       # (itself supervised, unlike ordinary tail padding)
model/transformer.py  # TransformerLM: RoPE, RMSNorm, SwiGLU MLP, weight-tied embeddings/head,
                       # GQA, hand-rolled causal attention via F.scaled_dot_product_attention with
                       # is_causal=(seq_len > 1) — SDPA's non-square causal bias is top-left-aligned,
                       # not cached-decode-safe, so masking is skipped for single-token cached decode
                       # instead — and a KV-cache-aware forward (position_offset/cache/layer_idx threaded through)
model/cache.py         # KVCache: per-layer (k, v) tensor cache, update() concatenates along seq dim
training/config.py    # DataConfig/ModelConfig/TrainConfig/GenerationConfig dataclasses, no YAML layer
training/train.py      # select_device, next_token_loss, make_collate_fn, train(), main() — the training entry point
training/checkpoint.py # saves/loads model + optimizer + dataset iterator state + model architecture config as one unit;
                       # optimizer is optional on load (inference callers pass none — no reason for
                       # generate.py to be coupled to train()'s optimizer shape, e.g. its param-group
                       # structure) and prune_old_checkpoints() keeps only the N most recent step_*.pt
                       # files (TrainConfig.keep_last_n_checkpoints, default 3) — checkpoints are ~3GB
                       # each at the real fineweb_edu-scale config, so unbounded accumulation over a
                       # long run is a real network-volume storage cost, not just a tidiness concern.
                       # save_checkpoint() writes to a step_N.pt.tmp file and only os.replace()s it
                       # into place on success, retrying transient failures (RuntimeError/OSError) up
                       # to 3 times with a 5s delay before giving up — added after a real run twice hit
                       # `RuntimeError: basic_ios::clear: iostream error` mid-torch.save against a
                       # network-mounted (MooseFS) --checkpoint-dir, once from a dropped SSH session and
                       # once under nohup/disown with no SSH involvement at all, pointing at transient
                       # network-volume write failures rather than anything SSH-related — see
                       # docs/training-guide.md Part 5
generate.py             # KV-cache-backed text generation from a checkpoint; main() is a second CLI entry point
s3.py                   # resolve_local_path()/sibling_path(): --checkpoint/--tokenizer-path accept
                       # s3://bucket/key as well as local paths (added so generate.py can run
                       # directly against a stopped pod's network volume without a manual scp first —
                       # RunPod exposes network volumes over an S3-compatible API even when the pod
                       # itself isn't running). boto3 is a lazy import inside resolve_local_path(), only
                       # required when an s3:// path is actually used (optional `s3` extra in
                       # pyproject.toml, same pattern as the `cuda` extra for liger-kernel). Downloads
                       # are cached under ~/.cache/llmtrain/s3/<bucket>/<key> keyed by bucket/key, skipped
                       # on repeat runs against the same checkpoint since checkpoints are immutable once
                       # written — matters in practice, checkpoints are ~1GB+. Endpoint/region come from
                       # the AWS_ENDPOINT_URL_S3/AWS_DEFAULT_REGION env vars (botocore >=1.31 resolves
                       # these automatically), so a RunPod S3 API key pair plus those two vars in .env is
                       # enough — no endpoint config in code.
logging_config.py       # dictConfig: stdout + JSONL file handler
```

```
python -m llmtrain.training.train --dataset <tiny_shakespeare|reformer_enwik8|fineweb_edu|smoltalk|no_robots> \
    [--max-steps N] [--batch-size N] [--lr F] [--checkpoint-dir DIR] [--resume PATH] \
    [--init-from-checkpoint PATH] [--tokenizer-path PATH]
```

Same code path for local smoke tests, the A100 smoke test, and the real pretraining run — `--checkpoint-dir` is the only thing that needs to change for RunPod (point it at the mounted network volume). Every `DataConfig`/`ModelConfig`/`TrainConfig` field is exposed as a CLI flag (`--d-model`, `--n-layers`, `--n-kv-heads`, `--dropout`, `--rope-theta`, `--min-lr`, `--warmup-steps`, `--weight-decay`, `--beta1`/`--beta2`, `--gradient-accumulation-steps`, `--grad-clip`, `--seed`, `--eval-interval`, `--keep-last-n-checkpoints`, `--compile`/`--no-compile`, `--use-amp`/`--no-use-amp`, `--use-fused-ce`/`--no-use-fused-ce`, `--wandb-project`, `--wandb-mode`, etc. — run `--help` for the full list); each flag's default reads from the corresponding dataclass field (e.g. `default=TrainConfig.checkpoint_interval`) rather than a duplicated literal, so the dataclasses are the single source of truth for defaults. `--init-from-checkpoint PATH` initializes model weights only (no optimizer/step state) from a pretrained checkpoint — how an SFT run starts from a pretraining checkpoint — and is mutually exclusive with `--resume`; `--tokenizer-path PATH` overrides the tokenizer loaded alongside it (defaults to `tokenizer.json` next to the checkpoint) and `main()` rejects `--tokenizer-path` without `--init-from-checkpoint` since it's otherwise silently ignored.

**Footgun:** unlike `--init-from-checkpoint`, `--resume` rebuilds the model from CLI-provided/default `ModelConfig` fields rather than the checkpoint's persisted architecture — so resuming an SFT run started via `--init-from-checkpoint` requires passing matching `--d-model`/`--n-layers`/etc. flags on the `--resume` invocation, or `load_state_dict` will raise a shape-mismatch error. This is a real, confirmed issue (a smoke test hit this exact architecture divergence); it's deliberate, documented behavior, not a bug to fix (`--resume`'s behavior is intentionally left untouched per the design spec). Separately, `DataConfig.max_seq_len` is **not** part of the checkpoint's persisted `model_config` and is not auto-adopted on `--init-from-checkpoint` — an SFT run can silently use a different `max_seq_len` than pretraining did unless the user passes a matching `--max-seq-len` flag.

`train.py`'s `get_lr(step, train_cfg)` is a linear-warmup-then-cosine-decay schedule (nanoGPT-style): ramps from 0 to `lr` over `warmup_steps`, cosine-decays to `min_lr` by `max_steps`, then holds at `min_lr`. It's a pure function of `step`, so `--resume` needs no separate scheduler state — the restored step counter alone determines the LR. The optimizer's `param_groups` are updated every step (before `optimizer.step()`), and the resulting LR is logged to W&B alongside loss. **`step` means an optimizer step, not a micro-batch forward/backward** — `train()` accumulates gradients over `TrainConfig.gradient_accumulation_steps` (default 8) micro-batches before each `optimizer.step()`/`step += 1`, so `--max-steps`, `checkpoint_interval`, and W&B's `step` axis all count optimizer steps; `--resume` always lands on a clean accumulation-window boundary, so no accumulation state needs to be persisted. Gradients are clipped (`torch.nn.utils.clip_grad_norm_`, `TrainConfig.grad_clip`, default 1.0) once per window, right before `optimizer.step()`; the pre-clip norm is logged to W&B as `grad_norm`. AdamW uses two param groups — `weight_decay` (default 0.1) applies only to parameters with `dim() >= 2`, excluding `nn.RMSNorm` gains (the only remaining 1-D params, since every `nn.Linear` layer is `bias=False`); `beta2` defaults to `0.95` (not PyTorch's `0.999`), matching GPT-3/LLaMA-style pretraining practice. `TransformerLM`'s weights use LLaMA-style init (`_init_weights`: `N(0, 0.02²)` for `nn.Linear`/`nn.Embedding`, with `attn.out_proj`/`mlp.w_down` additionally scaled by `1/√(2·n_layers)` since they write directly into the residual stream).

Three more cheap, safe additions live in `train()`: `torch.set_float32_matmul_precision("high")` gated behind `device.type == "cuda"` (TF32 matmul speedup on Ampere+, no-op on MPS/CPU); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set via `os.environ.setdefault` before any CUDA allocation (reduces fragmentation-driven OOMs on long runs); and `drop_last=True` on the `DataLoader` (keeps every batch shape constant so `torch.compile` never recompiles mid-run on a ragged final batch).

Deferred (discussed, not implemented — revisit only if the problem they solve actually shows up): sequence packing (concat + chunk instead of pad — real throughput win, but needs reworking the tokenizer/collate pipeline, not a config tweak), DataLoader `num_workers`/`persistent_workers`/`prefetch_factor` tuning, and gradient checkpointing — the last one solves memory pressure fused cross-entropy (now implemented) doesn't fully eliminate on its own.

Fused cross-entropy is implemented: `TrainConfig.use_fused_ce` (default `True`) routes `compute_loss()` through `next_token_loss_fused()` — a Liger-Kernel fused linear+cross-entropy Triton kernel — whenever `device.type == "cuda"`, avoiding materialization of the full `[batch*seq, vocab]` logits tensor (~4.3GB at current defaults). `TransformerLM.forward` gained `return_hidden: bool = False` to support this (returns post-`ln_f` hidden states instead of running through `self.head`; default-inert, `generate.py` is unaffected). On MPS/CPU, or with `--no-use-fused-ce`, `compute_loss()` falls back to the original full-logits `next_token_loss()` path. `liger-kernel` is an optional dependency (`pyproject.toml`'s `[project.optional-dependencies] cuda` group) — see the RunPod workflow section below for the install step.

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
- `model` in `train()` is explicitly annotated `torch.nn.Module` for consistency with `compute_loss()`/`evaluate()`'s parameter type. `train()` calls `model.compile()` (in-place, the current PyTorch-recommended pattern over functional `torch.compile(model)` wrapping) rather than reassigning `model` — no `OptimizedModule` wrapper is ever created, so there's no `_orig_mod.`-prefixed `state_dict()` key concern for `generate.py` (which always loads checkpoints into a fresh, uncompiled model) to worry about.

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
- `uv sync --extra cuda` once per pod to install `liger-kernel` — `TrainConfig.use_fused_ce` defaults to `True`, so training will `ImportError` partway into a run (after tokenizer training and dataset streaming) on a CUDA box that skips this step.
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

Everything except the GPU training loop itself gets a real, fast, CPU-only unit test with tiny fake data (no GPU, no network, no cost) — data loading, tokenizer, model forward/backward shapes, checkpoint round-trip (including dataset iterator state), config parsing. `train()`/`main()` orchestration has no automated test by design — it's validated by the manual smoke test documented in `docs/training-guide.md` (Part 1), most recently re-run end-to-end after the checkpoint/optimizer, checkpoint-pruning, and `model.compile()` fixes (30 steps, tiny_shakespeare, decreasing `val_loss`, valid pruned checkpoints, successful `generate.py` load).

## Development principles

- **Fail-fast TDD**: write a failing test before writing the implementation; keep feedback loops short, especially given GPU rental costs make late-discovered bugs expensive.
- **SOLID**: tokenizer, dataset loading, model, training loop, and evaluation are separable, substitutable concerns.
- **Karpathy principles for overengineering**: before adding abstraction, ask whether it earns its keep at the current scale of this toy project. Default to the simplest thing that works; prefer readable, hackable, single-purpose scripts over premature generalization. When in doubt about whether a component is overengineered, evaluate it against this standard rather than adding configurability "just in case."
