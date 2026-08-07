# Fused Cross-Entropy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Avoid materializing the full `[batch*seq, vocab]` logits tensor during training on CUDA (~4.3GB at current defaults) by fusing the final linear projection and the cross-entropy loss into one Liger-Kernel Triton computation, while leaving MPS/CPU and `generate.py` on the existing full-logits path unconditionally.

**Architecture:** Four small, mostly-independent additions followed by one integration task. `TransformerLM.forward` gains a `return_hidden` escape hatch (non-breaking, default-inert). `TrainConfig` gains a `use_fused_ce` toggle. `train.py` gains two new, not-yet-called functions (`next_token_loss_fused`, `compute_loss`) that depend on the first two. The final task wires `compute_loss()` into both `train()`'s loop and `evaluate()` (from the validation-loop plan), replacing their direct `next_token_loss(model(input_ids), ...)` calls — this changes `evaluate()`'s signature, so it's the one task that must also update `evaluate()`'s existing tests.

**Tech Stack:** PyTorch >=2.6, Liger Kernel (`liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`, CUDA/Triton-only, added as an optional dependency group), pytest.

## Global Constraints

- Every new `TrainConfig` field gets a corresponding `--flag` in `main()`, `default=TrainConfig.<field>` — wired in the task that consumes the field (Task 4 for `use_fused_ce`), not the task that defines it (Task 2), same convention used in both prior plans.
- Tests are CPU-only, tiny fake models/data, no GPU and no network — existing project convention. The actual fused Liger Kernel path is CUDA/Triton-only and cannot be exercised in this project's test suite or on local Mac dev at all — it's validated by a manual smoke test on the real A100 run, not by anything in this plan. Every test in this plan instead verifies the *non-fused* path and the *interface* the fused path depends on.
- `liger-kernel` is an **optional** dependency (a new `[project.optional-dependencies] cuda = [...]` group in `pyproject.toml`, not added to core `dependencies`) and is imported **lazily**, inside the one function that needs it — local Mac/CPU dev never needs it installed. `pyproject.toml` already sets `[tool.mypy] ignore_missing_imports = true`, so `uv run mypy src/` does not need any additional configuration to tolerate the (locally absent) `liger_kernel` import.
- `train()`/`main()` orchestration itself is not unit-tested by design (existing convention). Task 4's loop/`evaluate()` wiring is verified by the full test suite plus a manual CLI smoke run — necessarily on the non-fused path (`--no-use-fused-ce`, or simply run on a non-CUDA machine where it's auto-disabled), since no CUDA is available locally.
- Run `uv run pytest`, `uv run ruff check .`, and `uv run mypy src/` before each commit — the `pretraining-loop-hardening` plan's final review caught a real mypy failure that slipped through a task-level review, so this is not optional.

---

## Task 1: `TransformerLM.forward` gains `return_hidden`

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Produces: `TransformerLM.forward(input_ids, cache=None, return_hidden: bool = False) -> torch.Tensor` — when `True`, returns post-`ln_f` hidden states `[B, S, d_model]` instead of running through `self.head`. Consumed by Task 3's `compute_loss()`.
- Default (`return_hidden=False`) behavior is byte-for-byte unchanged — every existing caller (`generate.py`, `train()`'s current non-fused path, all existing tests) is unaffected. The full test suite must still pass unchanged after this task.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transformer.py`:

```python
def test_return_hidden_is_mathematically_inert():
    model = TransformerLM(_tiny_config())
    model.eval()
    input_ids = torch.randint(0, 16, (2, 6))
    with torch.no_grad():
        logits = model(input_ids)
        hidden = model(input_ids, return_hidden=True)
        logits_from_hidden = model.head(hidden)
    assert torch.allclose(logits, logits_from_hidden, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py::test_return_hidden_is_mathematically_inert -v`
Expected: FAIL with `TypeError: TransformerLM.forward() got an unexpected keyword argument 'return_hidden'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, replace `TransformerLM.forward`:

```python
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        x = self.token_emb(input_ids)
        position_offset = cache.seq_len if cache is not None else 0
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, position_offset=position_offset, cache=cache, layer_idx=layer_idx)
        x = self.ln_f(x)
        if return_hidden:
            return x
        return self.head(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS — all tests in the file, including every pre-existing test that calls `model(input_ids)` without `return_hidden` (default behavior unchanged).

Also run: `uv run pytest -q` (full suite) and `uv run mypy src/` to confirm nothing elsewhere broke — `generate.py` calls `model(input_ids, cache=...)` positionally/by-keyword without `return_hidden`, so it's unaffected, but confirm this with the full run rather than assuming.

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: add return_hidden to TransformerLM.forward"
```

---

## Task 2: `TrainConfig` gains `use_fused_ce`

**Files:**
- Modify: `src/llmtrain/training/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `TrainConfig.use_fused_ce: bool` (default `True`) — consumed by Task 4. No CLI flag yet (added in Task 4, per Global Constraints).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_train_config_defaults_to_fused_cross_entropy():
    cfg = TrainConfig()
    assert cfg.use_fused_ce is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_train_config_defaults_to_fused_cross_entropy -v`
Expected: FAIL with `AttributeError: 'TrainConfig' object has no attribute 'use_fused_ce'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/training/config.py`, add `use_fused_ce: bool = True` to `TrainConfig`, right after `use_amp: bool = True`:

```python
@dataclass
class TrainConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    grad_clip: float = 1.0
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_steps: int = 10000
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 125
    eval_interval: int = 500
    compile: bool = True
    use_amp: bool = True
    use_fused_ce: bool = True
    wandb_project: str = "llm-training"
    wandb_mode: str = "online"
    log_file: str = "app.log"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/config.py tests/test_config.py
git commit -m "feat: add use_fused_ce to TrainConfig"
```

---

## Task 3: `next_token_loss_fused()` + `compute_loss()` in `train.py`

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Modify: `pyproject.toml`
- Test: `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `TransformerLM.forward(..., return_hidden=True)` (Task 1).
- Produces: `compute_loss(model: torch.nn.Module, input_ids: torch.Tensor, pad_id: int, use_fused_ce: bool) -> torch.Tensor` — consumed by Task 4, which wires it into `train()`'s loop and `evaluate()`.
- This task adds two new, unused-so-far functions. It does not change `train()`'s or `evaluate()`'s existing behavior or call sites — the full test suite must still pass unchanged after this task.

- [ ] **Step 1: Write the failing test**

`tests/test_train_helpers.py` needs `compute_loss` added to its existing import from `llmtrain.training.train`:

```python
from llmtrain.training.train import (
    compute_loss,
    evaluate,
    get_lr,
    make_collate_fn,
    next_token_loss,
    select_device,
)
```

Add:

```python
def test_compute_loss_non_fused_matches_direct_next_token_loss():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, 16, (2, 6))

    with torch.no_grad():
        loss_via_compute_loss = compute_loss(model, input_ids, pad_id=0, use_fused_ce=False)
        logits = model(input_ids)
        loss_direct = next_token_loss(logits, input_ids, pad_id=0)

    assert torch.allclose(loss_via_compute_loss, loss_direct, atol=1e-6)
```

This test only exercises `use_fused_ce=False` — the fused branch needs CUDA/Liger Kernel and cannot be tested here, per the plan's Global Constraints.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_helpers.py::test_compute_loss_non_fused_matches_direct_next_token_loss -v`
Expected: FAIL with `ImportError: cannot import name 'compute_loss' from 'llmtrain.training.train'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/training/train.py`, add these two functions immediately after `next_token_loss` and before `evaluate()`:

```python
def next_token_loss_fused(
    hidden: torch.Tensor, head_weight: torch.Tensor, input_ids: torch.Tensor, pad_id: int
) -> torch.Tensor:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    shift_targets = input_ids[:, 1:].reshape(-1)
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=pad_id)
    return loss_fn(head_weight, shift_hidden, shift_targets)


def compute_loss(
    model: torch.nn.Module, input_ids: torch.Tensor, pad_id: int, use_fused_ce: bool
) -> torch.Tensor:
    if use_fused_ce:
        hidden = model(input_ids, return_hidden=True)
        head_weight = model.token_emb.weight
        return next_token_loss_fused(hidden, head_weight, input_ids, pad_id)
    logits = model(input_ids)
    return next_token_loss(logits, input_ids, pad_id)
```

The `from liger_kernel.transformers import ...` line is deliberately inside the function body, not at module level — this is what makes it a *lazy* import. `next_token_loss_fused` is only ever called from `compute_loss`'s `use_fused_ce=True` branch, and Task 4 will ensure `use_fused_ce` is only ever `True` when `device.type == "cuda"` — so on Mac/CPU dev, this line never executes and `liger-kernel` never needs to be installed.

Add the optional dependency group in `pyproject.toml`, right after the `dependencies = [...]` list in the `[project]` section:

```toml
[project]
name = "llmtrain"
version = "0.1.0"
description = "Toy LLM training pipeline"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.6",
    "tokenizers>=0.20",
    "datasets>=3.0",
    "wandb>=0.18",
]

[project.optional-dependencies]
cuda = ["liger-kernel"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: PASS — all tests in the file.

Also run: `uv run pytest -q` (full suite) and `uv run mypy src/` — both must be clean. Do NOT run `uv sync --extra cuda` or attempt to install `liger-kernel` locally — this task's tests never exercise the fused branch, so the dependency should remain uninstalled in local dev.

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py pyproject.toml
git commit -m "feat: add next_token_loss_fused and compute_loss, liger-kernel optional dep"
```

---

## Task 4: Wire `compute_loss()` into `train()` and `evaluate()`

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Modify: `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `TrainConfig.use_fused_ce` (Task 2), `compute_loss()` (Task 3).
- Produces: `evaluate()`'s signature changes — gains a new `use_fused_ce: bool` parameter (positional, appended after `use_amp`). This is a breaking change to `evaluate()`'s call contract, landing in the same commit as the fix to its only two callers: `train()`'s loop and this task's own updated tests. `train()` gains no new public interface — `use_fused_ce_effective` is a local variable, not a new parameter.

- [ ] **Step 1: Update `evaluate()`'s existing tests for the new signature**

In `tests/test_train_helpers.py`, both existing `evaluate()` tests currently call it with 6 positional arguments ending in `use_amp`. Update both call sites to add a 7th argument, `use_fused_ce=False`:

```python
def test_evaluate_returns_finite_float_and_restores_train_mode():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.train()

    batch = torch.randint(0, 16, (2, 6))
    val_dataloader = [batch, batch]

    val_loss = evaluate(
        model,
        val_dataloader,
        pad_id=0,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
        use_fused_ce=False,
    )

    assert math.isfinite(val_loss)
    assert model.training is True


def test_evaluate_restores_eval_mode_if_model_was_already_in_eval_mode():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.eval()

    batch = torch.randint(0, 16, (2, 6))
    val_dataloader = [batch]

    evaluate(
        model,
        val_dataloader,
        pad_id=0,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
        use_fused_ce=False,
    )

    assert model.training is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_helpers.py -k evaluate -v`
Expected: FAIL with `TypeError: evaluate() got an unexpected keyword argument 'use_fused_ce'` — `evaluate()` doesn't accept that parameter yet.

- [ ] **Step 3: Update `evaluate()`'s implementation**

In `src/llmtrain/training/train.py`, replace `evaluate()`:

```python
def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    pad_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    use_amp: bool,
    use_fused_ce: bool,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                losses.append(compute_loss(model, input_ids, pad_id, use_fused_ce).item())
    model.train(was_training)
    return sum(losses) / len(losses)
```

- [ ] **Step 4: Run the updated tests to verify they pass**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 5: Wire `compute_loss()` into `train()`'s loop and its call to `evaluate()`**

In `src/llmtrain/training/train.py`'s `train()` function:

Add a new local variable right after the existing `autocast_dtype = torch.bfloat16 if device.type == "cuda" else None` line:

```python
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    use_fused_ce_effective = train_cfg.use_fused_ce and device.type == "cuda"
```

Replace the loss-computation block inside the training loop:

```python
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=train_cfg.use_amp
            ):
                logits = model(input_ids)
                loss = (
                    next_token_loss(logits, input_ids, pad_id)
                    / train_cfg.gradient_accumulation_steps
                )
```

with:

```python
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=train_cfg.use_amp
            ):
                loss = (
                    compute_loss(model, input_ids, pad_id, use_fused_ce_effective)
                    / train_cfg.gradient_accumulation_steps
                )
```

Update the `evaluate()` call site to pass the new argument:

```python
            if step % train_cfg.eval_interval == 0:
                val_loss = evaluate(
                    model,
                    val_dataloader,
                    pad_id,
                    device,
                    autocast_dtype,
                    train_cfg.use_amp,
                    use_fused_ce_effective,
                )
```

- [ ] **Step 6: Add the `--use-fused-ce` CLI flag**

In `main()`, add next to the existing `--use-amp` argument:

```python
    parser.add_argument(
        "--use-fused-ce", action=argparse.BooleanOptionalAction, default=TrainConfig.use_fused_ce
    )
```

And thread it into the `TrainConfig(...)` construction, next to `use_amp=args.use_amp,`:

```python
        use_amp=args.use_amp,
        use_fused_ce=args.use_fused_ce,
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — every test in the project.

Also run: `uv run mypy src/` and `uv run ruff check .` — both must be clean.

- [ ] **Step 8: Manual smoke run to confirm the loop actually executes end to end**

Run:
```bash
uv run python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps 4 \
  --gradient-accumulation-steps 2 --batch-size 2 --checkpoint-interval 2 --eval-interval 2 \
  --wandb-mode disabled
```
Expected: completes without error, identical output to the pre-Task-4 baseline (val_loss entries in the JSONL log at steps 2/4, checkpoints written) — this exercises the non-fused path only, since local dev has no CUDA, `use_fused_ce_effective` evaluates to `False` regardless of the `--use-fused-ce` default, and the loop should behave exactly as it did before this task. This confirms the wiring didn't break anything; it does NOT and cannot confirm the fused kernel itself works — that's validated on the real A100 run, outside this plan's scope.

- [ ] **Step 9: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py
git commit -m "feat: wire compute_loss into train() and evaluate(), add --use-fused-ce"
```

---

## Self-Review Notes

- **Spec coverage:** All four `## Components` sections of `2026-08-06-fused-cross-entropy-design.md` map to a task — `return_hidden` → Task 1, config → Task 2, `next_token_loss_fused`/`compute_loss` → Task 3, wiring + dependency → Tasks 3 (dependency) and 4 (wiring). The spec's `generate.py`-untouched decision required no task (explicit non-change, verified in Task 1's Step 4 full-suite run).
- **Type consistency:** `compute_loss(model, input_ids, pad_id, use_fused_ce) -> torch.Tensor` is defined once (Task 3) and called identically in Task 4 (both inside `train()`'s loop and inside the rewritten `evaluate()`). `evaluate()`'s new `use_fused_ce: bool` parameter is added consistently to its definition, both of its pre-existing tests, and its one call site — all in Task 4, so no task boundary leaves a signature mismatch.
- **Breaking-change containment:** `evaluate()`'s signature change (Task 4) and `TransformerLM.forward`'s new optional parameter (Task 1, non-breaking) are the only two interface changes in this plan. The `evaluate()` break is deliberately contained to a single task, alongside both places that call it, matching how the `validation-loop` plan's own breaking rename (`load_streaming_dataset` → `load_streaming_datasets`) was kept atomic with its call site.
- **Placeholder scan:** no TBD/TODO; every step includes literal code, not a description of code.
