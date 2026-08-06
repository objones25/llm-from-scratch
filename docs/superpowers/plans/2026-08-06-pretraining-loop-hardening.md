# Pretraining-Loop Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `train()` behave like a real LLM pretraining recipe instead of a bare forward/backward loop: gradient accumulation (with a step-semantics redefinition to optimizer-step), gradient clipping with grad-norm observability, retuned AdamW hyperparameters, and LLaMA-style weight initialization with bias removal.

**Architecture:** Four independent-but-sequenced changes. `TrainConfig` (`training/config.py`) gains two new fields and two changed defaults — a pure config change other tasks build on. `TransformerLM` (`model/transformer.py`) gets bias removal + a new `_init_weights` function — self-contained, touches no other file. `train()`'s loop (`training/train.py`) is restructured in two steps: first to accumulate gradients over micro-batches before stepping the optimizer (redefining what "step" means), then to clip gradients and log their norm at that same window boundary. No other files change.

**Tech Stack:** PyTorch >=2.6 (`torch.nn.utils.clip_grad_norm_`, `torch.nn.init.normal_`, `nn.Module.apply`), pytest.

## Global Constraints

- Every new `TrainConfig` field gets a corresponding `--flag` in `main()`, with `default=TrainConfig.<field>` (never a duplicated literal) — existing project convention.
- Tests are CPU-only, tiny fake models/data, no GPU and no network — existing project convention (CLAUDE.md testing strategy).
- `train()`/`main()` orchestration itself is not unit-tested by design (existing convention). Loop changes in Tasks 3–4 are verified by a quick manual CLI smoke run instead — `docs/smoke-test.md` was deleted this session, so there's no existing doc to follow; the exact command to run is given in each task.
- `gradient_accumulation_steps` and `grad_clip` are trusted config inputs — do not add validation for them, matching the project's existing convention for internal/trusted config.
- `nn.RMSNorm`'s learnable gain stays at PyTorch's default (ones) — not touched by the weight-init work.
- Run `uv run pytest`, `uv run ruff check .`, and `uv run mypy src/` before each commit.

---

## Task 1: `TrainConfig` gains accumulation/clipping fields and retuned AdamW defaults

**Files:**
- Modify: `src/llmtrain/training/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `TrainConfig.gradient_accumulation_steps: int` (default `8`), `TrainConfig.grad_clip: float` (default `1.0`), `TrainConfig.beta2` (default `0.95`), `TrainConfig.weight_decay` (default `0.1`) — consumed by Task 3 (`gradient_accumulation_steps`) and Task 4 (`grad_clip`).

- [ ] **Step 1: Write the failing test**

`tests/test_config.py` doesn't import `pytest` yet. Add it as the first line of the file:

```python
import pytest

from llmtrain.training.config import DataConfig, GenerationConfig, ModelConfig, TrainConfig
```

Then add to `tests/test_config.py`:

```python
def test_train_config_has_gradient_accumulation_and_clip_defaults():
    cfg = TrainConfig()
    assert cfg.gradient_accumulation_steps >= 1
    assert cfg.grad_clip > 0


def test_train_config_uses_llm_pretraining_adamw_values():
    cfg = TrainConfig()
    assert cfg.beta2 == pytest.approx(0.95)
    assert cfg.weight_decay == pytest.approx(0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k "accumulation_and_clip or adamw_values" -v`
Expected: FAIL — `test_train_config_has_gradient_accumulation_and_clip_defaults` fails with `TypeError`/`AttributeError` (fields don't exist yet); `test_train_config_uses_llm_pretraining_adamw_values` fails because `beta2`/`weight_decay` are still `0.999`/`0.01`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/training/config.py`, replace the `TrainConfig` dataclass:

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
    checkpoint_interval: int = 1000
    compile: bool = True
    use_amp: bool = True
    wandb_project: str = "llm-training"
    wandb_mode: str = "online"
    log_file: str = "app.log"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all tests in the file, including the pre-existing `test_train_config_has_sensible_defaults` which only asserts ranges, not exact values, so it's unaffected by the beta2/weight_decay change).

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/config.py tests/test_config.py
git commit -m "feat: add gradient accumulation/clipping config, retune AdamW defaults"
```

---

## Task 2: `TransformerLM` — remove biases, add LLaMA-style weight init

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Independent of the other three tasks — touches no shared state they rely on.

- [ ] **Step 1: Write the failing tests**

`tests/test_transformer.py` already imports `pytest` (line 1). Add to `tests/test_transformer.py`:

```python
def test_linear_layers_have_no_bias():
    model = TransformerLM(_tiny_config())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None, f"{name} has a bias, expected bias=False"


def test_token_embedding_weight_std_is_approximately_0_02():
    config = ModelConfig(
        vocab_size=4000, d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0
    )
    model = TransformerLM(config)
    assert model.token_emb.weight.std().item() == pytest.approx(0.02, abs=0.002)


def test_residual_projections_are_scaled_down_from_plain_init():
    config = ModelConfig(
        vocab_size=4000, d_model=64, n_layers=6, n_heads=2, n_kv_heads=1, dropout=0.0
    )
    model = TransformerLM(config)
    expected_std = 0.02 / (2 * config.n_layers) ** 0.5
    out_proj_std = model.blocks[0].attn.out_proj.weight.std().item()
    w_down_std = model.blocks[0].mlp.w_down.weight.std().item()
    assert out_proj_std == pytest.approx(expected_std, rel=0.4)
    assert w_down_std == pytest.approx(expected_std, rel=0.4)
    assert out_proj_std < 0.02
    assert w_down_std < 0.02
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transformer.py -k "no_bias or embedding_weight_std or residual_projections_are_scaled" -v`
Expected: FAIL — `test_linear_layers_have_no_bias` fails because every `nn.Linear` currently has a default bias; the two std tests fail because weights currently use PyTorch's default `kaiming_uniform_` init, not `N(0, 0.02²)`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, add `bias=False` to every `nn.Linear` in `CausalSelfAttention.__init__`:

```python
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.d_model, 2 * config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
```

And in `MLP.__init__`:

```python
        self.w_gate = nn.Linear(config.d_model, d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, config.d_model, bias=False)
```

Add a module-level `_init_weights` function right after `apply_rotary` (before `class CausalSelfAttention`):

```python
@torch.no_grad()
def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

In `TransformerLM.__init__`, apply it and rescale the residual-stream output projections after weight tying:

```python
class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(_init_weights)
        residual_std = 0.02 / (2 * config.n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=residual_std)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS — all tests in the file, including pre-existing ones (bias removal and re-init don't change any tensor shape, so shape/gradient/weight-tying tests are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: remove linear biases, add LLaMA-style weight init"
```

---

## Task 3: Gradient accumulation in the training loop

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Test: `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `TrainConfig.gradient_accumulation_steps` (Task 1).
- Produces: `train()`'s loop now treats `step` as an optimizer step (post-accumulation) — Task 4 inserts gradient clipping into this same restructured loop, right before the `optimizer.step()` call this task establishes.

- [ ] **Step 1: Add a reference test locking in the accumulation math**

This test doesn't call any new `llmtrain` code — it's a standalone property test of the exact gradient-accumulation math the loop is about to implement (divide loss by N, call `.backward()` N times, one `optimizer.step()`), using a tiny `torch.nn.Linear` model with a uniform micro-batch size (matching the project's `drop_last=True` DataLoader, which guarantees every micro-batch is the same size). It passes immediately — that's expected, not a bug — and exists to document and lock in the semantics `train()`'s loop is about to rely on, before touching orchestration code that isn't itself unit-tested.

Add to `tests/test_train_helpers.py`:

```python
def test_gradient_accumulation_matches_full_batch_gradient():
    torch.manual_seed(0)
    model_full = torch.nn.Linear(4, 2)
    model_accum = torch.nn.Linear(4, 2)
    model_accum.load_state_dict(model_full.state_dict())

    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    loss_full = torch.nn.functional.mse_loss(model_full(x), y)
    loss_full.backward()

    accumulation_steps = 4
    micro_batch_size = 2
    for i in range(accumulation_steps):
        start, end = i * micro_batch_size, (i + 1) * micro_batch_size
        micro_loss = (
            torch.nn.functional.mse_loss(model_accum(x[start:end]), y[start:end])
            / accumulation_steps
        )
        micro_loss.backward()

    for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
        assert torch.allclose(p_full.grad, p_accum.grad, atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: Run the test to confirm it passes**

Run: `uv run pytest tests/test_train_helpers.py::test_gradient_accumulation_matches_full_batch_gradient -v`
Expected: PASS immediately (this locks in the target math; there's no new production code for it to fail against).

- [ ] **Step 3: Restructure `train()`'s loop to accumulate gradients**

In `src/llmtrain/training/train.py`, replace the training loop body (currently `model.train()` through `wandb.finish()`):

```python
    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    micro_step = 0
    while step < train_cfg.max_steps:
        for batch in dataloader:
            input_ids = batch.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=train_cfg.use_amp
            ):
                logits = model(input_ids)
                loss = (
                    next_token_loss(logits, input_ids, pad_id)
                    / train_cfg.gradient_accumulation_steps
                )

            loss.backward()
            accumulated_loss += loss.item()
            micro_step += 1

            if micro_step % train_cfg.gradient_accumulation_steps != 0:
                continue

            lr = get_lr(step, train_cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"loss": accumulated_loss, "lr": lr}, step=step)
            logger.debug("step %d complete", step, extra={"step": step})
            accumulated_loss = 0.0

            step += 1
            if step % train_cfg.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    step=step,
                    dataset_state=dataset.state_dict(),
                )
                logger.info("saved checkpoint at step %d", step, extra={"step": step})
            if step >= train_cfg.max_steps:
                break

    wandb.finish()
    logger.info("training complete after %d steps", step, extra={"step": step})
```

Note what changed: `optimizer.zero_grad()` moved from the top of the per-micro-batch loop to once before the loop starts and once after each `optimizer.step()`; the loss is divided by `gradient_accumulation_steps` before `.backward()`; the `if micro_step % ... != 0: continue` gate means `step`/checkpointing/W&B logging only happen once per full accumulation window. `micro_step` lives outside both the `while` and `for` loops, so a window that spans an epoch restart (relevant for small datasets like `tiny_shakespeare`) just keeps accumulating — no special-casing needed.

- [ ] **Step 4: Add the `--gradient-accumulation-steps` CLI flag**

In `main()`, add next to the existing `--batch-size` argument:

```python
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TrainConfig.gradient_accumulation_steps,
    )
```

And thread it into the `TrainConfig(...)` construction further down in `main()`:

```python
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        ...
```

Keep every other existing field in that call (`min_lr=args.min_lr,` through `log_file=args.log_file,`) unchanged — only the new `gradient_accumulation_steps=args.gradient_accumulation_steps,` line is inserted, right after `batch_size=args.batch_size,`.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — no existing test calls `train()`/`main()` directly (per the project's existing convention), so this is confirming nothing else broke.

- [ ] **Step 6: Manual smoke run to confirm the loop actually executes**

Run:
```bash
uv run python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps 4 \
  --gradient-accumulation-steps 2 --batch-size 2 --checkpoint-interval 2 --wandb-mode disabled
```
Expected: completes without error, logs 4 `step %d complete` lines (not 8 — confirms `step` is now counted post-accumulation), and writes `checkpoints/step_2.pt` and `checkpoints/step_4.pt`.

- [ ] **Step 7: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py
git commit -m "feat: implement gradient accumulation, redefine step as optimizer-step"
```

---

## Task 4: Gradient clipping + grad-norm logging

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Test: `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `TrainConfig.grad_clip` (Task 1); inserts into the accumulation-window boundary established in Task 3, right before `optimizer.step()`.

- [ ] **Step 1: Add a reference test locking in the clipping math**

Like Task 3's Step 1, this is a standalone property test of `torch.nn.utils.clip_grad_norm_` itself (not new `llmtrain` code) — it passes immediately and documents the exact behavior the loop is about to rely on: the function returns the pre-clip L2 norm, and scales gradients in-place so the post-clip norm never exceeds `max_norm`.

Add to `tests/test_train_helpers.py`:

```python
def test_clip_grad_norm_caps_gradient_norm_and_returns_pre_clip_norm():
    model = torch.nn.Linear(4, 2)
    x = torch.randn(3, 4)
    loss = (model(x) * 1000.0).sum()
    loss.backward()

    manual_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()))
    max_norm = 1.0
    returned_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    assert returned_norm.item() == pytest.approx(manual_norm.item(), rel=1e-4)
    post_clip_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()))
    assert post_clip_norm.item() <= max_norm + 1e-4
```

- [ ] **Step 2: Run the test to confirm it passes**

Run: `uv run pytest tests/test_train_helpers.py::test_clip_grad_norm_caps_gradient_norm_and_returns_pre_clip_norm -v`
Expected: PASS immediately.

- [ ] **Step 3: Wire clipping + grad-norm logging into the accumulation-window boundary**

In `src/llmtrain/training/train.py`, inside the `if micro_step % train_cfg.gradient_accumulation_steps != 0: continue` block from Task 3, replace this section:

```python
            lr = get_lr(step, train_cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"loss": accumulated_loss, "lr": lr}, step=step)
```

with:

```python
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            lr = get_lr(step, train_cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            wandb.log(
                {"loss": accumulated_loss, "lr": lr, "grad_norm": total_norm.item()}, step=step
            )
```

- [ ] **Step 4: Add the `--grad-clip` CLI flag**

In `main()`, add next to the new `--gradient-accumulation-steps` argument from Task 3:

```python
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
```

And thread it into the `TrainConfig(...)` construction:

```python
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip=args.grad_clip,
        lr=args.lr,
        ...
```

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS.

- [ ] **Step 6: Manual smoke run to confirm clipping doesn't break the loop**

Run:
```bash
uv run python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps 4 \
  --gradient-accumulation-steps 2 --batch-size 2 --checkpoint-interval 2 --wandb-mode disabled
```
Expected: same as Task 3's smoke run (completes, 4 steps logged, two checkpoints written) — confirms clipping doesn't crash or change the step count.

- [ ] **Step 7: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py
git commit -m "feat: add gradient clipping and grad-norm logging"
```

---

## Self-Review Notes

- **Spec coverage:** All four `## Components` sections of `2026-08-06-pretraining-loop-hardening-design.md` map to a task (accumulation → Task 3, clipping/grad-norm → Task 4, AdamW retune → Task 1, weight init/bias removal → Task 2). The spec's `warmup_steps`-stays-unchanged conclusion required no task (explicitly a non-change). The spec's "no new error paths" conclusion required no task.
- **Type consistency:** `gradient_accumulation_steps: int`, `grad_clip: float`, `beta2: float`, `weight_decay: float` are used identically across Task 1 (definition) and Tasks 3–4 (consumption). `_init_weights(module: nn.Module) -> None` signature matches its one call site (`self.apply(_init_weights)`).
- **Placeholder scan:** no TBD/TODO; every step includes literal code, not a description of code.
