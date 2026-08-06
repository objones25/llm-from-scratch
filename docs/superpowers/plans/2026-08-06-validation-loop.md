# Held-Out Validation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `train()` a held-out validation loss signal — currently it only ever logs train loss, so there's no way to notice overfitting or judge generalization during a long paid A100 run.

**Architecture:** Three independent, non-breaking additions (`DatasetSpec`'s per-dataset validation strategy, `TrainConfig.eval_interval`, a standalone `evaluate()` helper) followed by one integration task that renames `load_streaming_dataset` to `load_streaming_datasets` (a breaking return-type change) and wires everything into `train()`'s loop in the same commit — the rename and its one call site must move together, or the tree breaks between commits.

**Tech Stack:** PyTorch >=2.6, Hugging Face `datasets` (`IterableDataset.shuffle`/`.take`/`.skip`/`.state_dict`/`.load_state_dict`), pytest (incl. `monkeypatch.setitem` for registry-scoped test fixtures).

## Global Constraints

- Every new `TrainConfig` field eventually gets a corresponding `--flag` in `main()`, `default=TrainConfig.<field>` — but the flag is wired in the task that actually consumes the field, not necessarily the task that defines it (same convention used in the prior `pretraining-loop-hardening` plan: `--eval-interval` is added in Task 4, not Task 2).
- Tests are CPU-only, tiny fake data/models, no GPU and no network — existing project convention (CLAUDE.md testing strategy).
- `train()`/`main()` orchestration itself is not unit-tested by design (existing convention). Task 4's loop integration is verified by the full test suite plus a manual CLI smoke run.
- `eval_interval` is a trusted config value — no validation added for it, matching the existing treatment of `grad_clip`/`gradient_accumulation_steps`.
- `DatasetSpec.__post_init__`'s fail-fast `ValueError` on a malformed registry entry matches the project's existing validation style (see `CausalSelfAttention.__init__` in `model/transformer.py`).
- Run `uv run pytest`, `uv run ruff check .`, and `uv run mypy src/` before each commit — the prior plan's final review caught a real mypy failure that slipped through a task-level review, so this is not optional.

---

## Task 1: `DatasetSpec` gains a per-dataset validation strategy

**Files:**
- Modify: `src/llmtrain/data/streaming.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Produces: `DatasetSpec.val_split: str | None` (default `None`), `DatasetSpec.val_holdout_examples: int | None` (default `None`), with a `__post_init__` that raises `ValueError` unless exactly one is set. `DATASET_REGISTRY["tiny_shakespeare"]` gets `val_split="test"`; `DATASET_REGISTRY["reformer_enwik8"]` and `DATASET_REGISTRY["fineweb_edu"]` get `val_holdout_examples=1000`. Consumed by Task 4's `load_streaming_datasets`.
- This task does NOT touch `load_streaming_dataset` (the function keeps its current name and single-`IterableDataset` return type until Task 4) — the new `DatasetSpec` fields simply aren't read by anything yet. The full test suite must still pass unchanged after this task.

- [ ] **Step 1: Write the failing tests**

`tests/test_streaming.py` doesn't import `pytest` or `DatasetSpec` yet. Update its imports:

```python
import pytest
from datasets import Dataset

from llmtrain.data.streaming import DATASET_REGISTRY, DatasetSpec, load_streaming_dataset
```

Add:

```python
def test_dataset_spec_rejects_both_val_split_and_val_holdout_examples():
    with pytest.raises(ValueError):
        DatasetSpec(path="x", name=None, split="train", val_split="test", val_holdout_examples=5)


def test_dataset_spec_rejects_neither_val_split_nor_val_holdout_examples():
    with pytest.raises(ValueError):
        DatasetSpec(path="x", name=None, split="train")


def test_tiny_shakespeare_registry_entry_uses_native_val_split():
    spec = DATASET_REGISTRY["tiny_shakespeare"]
    assert spec.val_split == "test"
    assert spec.val_holdout_examples is None


def test_reformer_enwik8_and_fineweb_edu_registry_entries_carve_val_holdout():
    for name in ["reformer_enwik8", "fineweb_edu"]:
        spec = DATASET_REGISTRY[name]
        assert spec.val_split is None
        assert spec.val_holdout_examples == 1000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: `test_dataset_spec_rejects_*` fail with `TypeError: DatasetSpec.__init__() got an unexpected keyword argument 'val_split'` (the fields don't exist yet). `test_tiny_shakespeare_registry_entry_uses_native_val_split` and `test_reformer_enwik8_and_fineweb_edu_registry_entries_carve_val_holdout` fail with `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/data/streaming.py`, replace the `DatasetSpec` dataclass and `DATASET_REGISTRY`:

```python
@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    split: str
    text_column: str = "text"
    val_split: str | None = None
    val_holdout_examples: int | None = None

    def __post_init__(self) -> None:
        if (self.val_split is None) == (self.val_holdout_examples is None):
            raise ValueError("exactly one of val_split or val_holdout_examples must be set")


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tiny_shakespeare": DatasetSpec(
        path="Trelis/tiny-shakespeare",
        name=None,
        split="train",
        text_column="Text",
        val_split="test",
    ),
    "reformer_enwik8": DatasetSpec(
        path="reds0510/enwik8-processed", name=None, split="train", val_holdout_examples=1000
    ),
    "fineweb_edu": DatasetSpec(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        val_holdout_examples=1000,
    ),
}
```

Leave `load_streaming_dataset` (the function below `DATASET_REGISTRY`) completely unchanged for this task.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: PASS — all tests in the file, including the pre-existing `test_fineweb_edu_registry_entry_uses_sample_100bt_config` and `test_load_streaming_dataset_shuffles_and_yields_every_example` (unaffected, since `load_streaming_dataset` wasn't touched).

Also run: `uv run pytest -q` (full suite) to confirm nothing elsewhere broke, and `uv run mypy src/` to confirm the new `__post_init__` type-checks cleanly.

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/data/streaming.py tests/test_streaming.py
git commit -m "feat: add per-dataset validation strategy to DatasetSpec"
```

---

## Task 2: `TrainConfig` gains `eval_interval`

**Files:**
- Modify: `src/llmtrain/training/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `TrainConfig.eval_interval: int` (default `500`) — consumed by Task 4. No CLI flag yet (added in Task 4, per Global Constraints).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_train_config_has_eval_interval_default():
    cfg = TrainConfig()
    assert cfg.eval_interval > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_train_config_has_eval_interval_default -v`
Expected: FAIL with `AttributeError: 'TrainConfig' object has no attribute 'eval_interval'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/training/config.py`, add `eval_interval: int = 500` to `TrainConfig`, right after `checkpoint_interval: int = 125`:

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
git commit -m "feat: add eval_interval to TrainConfig"
```

---

## Task 3: `evaluate()` helper in `train.py`

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Test: `tests/test_train_helpers.py`

**Interfaces:**
- Produces: `evaluate(model: torch.nn.Module, val_dataloader, pad_id: int, device: torch.device, autocast_dtype: torch.dtype | None, use_amp: bool) -> float` — consumed by Task 4's `train()` loop.
- This task adds a new, unused-so-far function. It does not change `train()`'s existing behavior or call sites — the full test suite must still pass unchanged after this task.

- [ ] **Step 1: Write the failing tests**

`tests/test_train_helpers.py` needs `import math` added (not currently imported) and `evaluate` added to its existing import from `llmtrain.training.train`:

```python
import math

import pytest
import torch

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import ModelConfig, TrainConfig
from llmtrain.training.train import evaluate, get_lr, make_collate_fn, next_token_loss, select_device
```

Add:

```python
def test_evaluate_returns_finite_float_and_restores_train_mode():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.train()

    # A plain list of batches stands in for a DataLoader here — evaluate() only ever
    # does `for batch in val_dataloader:`, so any iterable of pre-batched tensors works,
    # and this avoids pulling in real DataLoader/dataset machinery for a unit test.
    batch = torch.randint(0, 16, (2, 6))
    val_dataloader = [batch, batch]

    val_loss = evaluate(
        model,
        val_dataloader,
        pad_id=0,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
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
    )

    assert model.training is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_helpers.py -k evaluate -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate' from 'llmtrain.training.train'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/training/train.py`, add this function after `next_token_loss` and before `make_collate_fn`:

```python
def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    pad_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    use_amp: bool,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                logits = model(input_ids)
                losses.append(next_token_loss(logits, input_ids, pad_id).item())
    model.train(was_training)
    return sum(losses) / len(losses)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: PASS — all tests in the file.

Also run: `uv run pytest -q` (full suite) and `uv run mypy src/` to confirm nothing broke.

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py
git commit -m "feat: add evaluate() helper for held-out validation loss"
```

---

## Task 4: `load_streaming_datasets` (renamed) + full `train()`/`main()` integration

**Files:**
- Modify: `src/llmtrain/data/streaming.py`
- Modify: `src/llmtrain/training/train.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Consumes: `DatasetSpec.val_split`/`val_holdout_examples` (Task 1), `TrainConfig.eval_interval` (Task 2), `evaluate()` (Task 3).
- Produces: `load_streaming_datasets(dataset_name, seed, buffer_size, load_fn=load_dataset) -> tuple[IterableDataset, IterableDataset]` (train, val) — replaces `load_streaming_dataset`, which is deleted. This is the plan's only breaking change, and it lands in the same commit as the one call site that uses it (`train()`), so the tree is never left in a broken intermediate state.

- [ ] **Step 1: Write the failing tests for the renamed/split-returning function**

In `tests/test_streaming.py`, update the import line (still named `load_streaming_dataset`, singular, as left by Task 1) to the new name:

```python
from llmtrain.data.streaming import DATASET_REGISTRY, DatasetSpec, load_streaming_datasets
```

Then replace the existing `test_load_streaming_dataset_shuffles_and_yields_every_example` and add three new tests:

```python
def test_load_streaming_datasets_shuffles_and_yields_every_train_example():
    train_dataset, _val_dataset = load_streaming_datasets(
        "tiny_shakespeare", seed=42, buffer_size=5, load_fn=_fake_load_dataset
    )
    examples = list(train_dataset)
    assert len(examples) == 20
    assert all("text" in example for example in examples)


def test_load_streaming_datasets_carve_path_splits_train_and_val(monkeypatch):
    # text_column="Text" matches what _fake_load_dataset actually returns (capital T) —
    # without it, spec.text_column defaults to "text" and the rename_column step that
    # normally handles this mismatch gets skipped, so example["text"] would KeyError.
    monkeypatch.setitem(
        DATASET_REGISTRY,
        "carve_test",
        DatasetSpec(
            path="x", name=None, split="train", text_column="Text", val_holdout_examples=5
        ),
    )
    train_dataset, val_dataset = load_streaming_datasets(
        "carve_test", seed=42, buffer_size=5, load_fn=_fake_load_dataset
    )
    val_examples = list(val_dataset)
    train_examples = list(train_dataset)
    assert len(val_examples) == 5
    assert len(train_examples) == 15
    val_texts = {example["text"] for example in val_examples}
    train_texts = {example["text"] for example in train_examples}
    assert val_texts.isdisjoint(train_texts)


def test_load_streaming_datasets_native_split_path_uses_val_split_name(monkeypatch):
    def _fake_load_dataset_by_split(path, name, split, streaming):
        texts = [f"{split}-example-{i}" for i in range(5)]
        return Dataset.from_dict({"text": texts}).to_iterable_dataset(num_shards=1)

    monkeypatch.setitem(
        DATASET_REGISTRY,
        "native_test",
        DatasetSpec(path="x", name=None, split="train", val_split="validation"),
    )
    _train_dataset, val_dataset = load_streaming_datasets(
        "native_test", seed=42, buffer_size=5, load_fn=_fake_load_dataset_by_split
    )
    val_texts = {example["text"] for example in val_dataset}
    assert val_texts == {f"validation-example-{i}" for i in range(5)}


def test_shuffled_skip_dataset_resumes_correctly_via_state_dict():
    # This doesn't call load_streaming_datasets directly — it's a standalone check of
    # the exact mechanism the carve path depends on (shuffle().skip(n) + state_dict()/
    # load_state_dict()), since the `datasets` library's own docs don't explicitly
    # confirm this combination round-trips correctly for exact --resume.
    def _build_source():
        return Dataset.from_dict(
            {"text": [f"example {i}" for i in range(20)]}
        ).to_iterable_dataset(num_shards=4)

    n = 5
    dataset = _build_source().shuffle(seed=42, buffer_size=5).skip(n)

    consumed = []
    state = None
    for idx, example in enumerate(dataset):
        consumed.append(example)
        if idx == 2:
            state = dataset.state_dict()
            break

    resumed_dataset = _build_source().shuffle(seed=42, buffer_size=5).skip(n)
    resumed_dataset.load_state_dict(state)
    remaining = list(resumed_dataset)

    full = list(_build_source().shuffle(seed=42, buffer_size=5).skip(n))

    assert consumed + remaining == full
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: the whole file FAILS TO COLLECT with `ImportError: cannot import name 'load_streaming_datasets' from 'llmtrain.data.streaming'` — the import line was updated in Step 1, but `streaming.py` itself still only defines `load_streaming_dataset` (singular) until Step 3, so every test in the file reports as a collection error, not just the four new ones. This is expected for this step; proceed to Step 3.

Note that `test_shuffled_skip_dataset_resumes_correctly_via_state_dict` doesn't actually need `load_streaming_datasets` at all (it only exercises the `datasets` library directly) — its real verification happens in Step 4, once the file collects successfully again. If it fails there, that's a real finding about the `datasets` library's resume behavior, not a bug in your implementation — stop and report it rather than trying to work around it.

- [ ] **Step 3: Rename and rewrite `load_streaming_dataset`**

In `src/llmtrain/data/streaming.py`, replace the function:

```python
def load_streaming_datasets(
    dataset_name: str,
    seed: int,
    buffer_size: int,
    load_fn: Callable[..., IterableDataset] = load_dataset,
) -> tuple[IterableDataset, IterableDataset]:
    spec = DATASET_REGISTRY[dataset_name]
    dataset = load_fn(spec.path, name=spec.name, split=spec.split, streaming=True)
    if spec.text_column != "text":
        dataset = dataset.rename_column(spec.text_column, "text")
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    if spec.val_split is not None:
        val_dataset = load_fn(spec.path, name=spec.name, split=spec.val_split, streaming=True)
        if spec.text_column != "text":
            val_dataset = val_dataset.rename_column(spec.text_column, "text")
        return shuffled, val_dataset

    val_dataset = shuffled.take(spec.val_holdout_examples)
    train_dataset = shuffled.skip(spec.val_holdout_examples)
    return train_dataset, val_dataset
```

- [ ] **Step 4: Run the streaming tests to verify they pass**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: PASS — all tests in the file. (The full suite will still fail at this point, because `train.py` still imports the now-deleted `load_streaming_dataset` — that's expected and fixed in the next step.)

- [ ] **Step 5: Update `train()`/`main()` to use `load_streaming_datasets`**

In `src/llmtrain/training/train.py`:

Change the import:

```python
from llmtrain.data.streaming import load_streaming_datasets
```

Replace the dataset-loading section near the top of `train()`:

```python
    train_dataset, val_dataset = load_streaming_datasets(
        data_cfg.dataset_name, seed=train_cfg.seed, buffer_size=data_cfg.shuffle_buffer_size
    )
    sample_texts = [
        example["text"] for example in train_dataset.take(data_cfg.tokenizer_sample_size)
    ]
    tokenizer = train_tokenizer(sample_texts, vocab_size=data_cfg.tokenizer_vocab_size)
    model_cfg.vocab_size = tokenizer.get_vocab_size()
```

Every remaining reference to the old `dataset` variable in `train()` renames to `train_dataset`:
- The resume block: `dataset.load_state_dict(dataset_state)` → `train_dataset.load_state_dict(dataset_state)` (only the `IterableDataset` object is renamed — `dataset_state`, the loaded checkpoint state dict, keeps its name).
- The `DataLoader(...)` construction: its first positional argument `dataset` → `train_dataset`.
- The checkpoint save call: `dataset_state=dataset.state_dict()` → `dataset_state=train_dataset.state_dict()`.

Immediately after the existing train `dataloader = DataLoader(...)` block, add a val dataloader:

```python
    val_dataloader = DataLoader(
        val_dataset,  # type: ignore[arg-type]  # IterableDataset isn't in DataLoader's stub overloads, but is supported at runtime
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len),
    )
```

In the training loop, immediately after the existing `if step % train_cfg.checkpoint_interval == 0: save_checkpoint(...)` block and before `if step >= train_cfg.max_steps: break`, add:

```python
            if step % train_cfg.eval_interval == 0:
                val_loss = evaluate(
                    model, val_dataloader, pad_id, device, autocast_dtype, train_cfg.use_amp
                )
                wandb.log({"val_loss": val_loss}, step=step)
                logger.info(
                    "val_loss %.4f at step %d",
                    val_loss,
                    step,
                    extra={"step": step, "val_loss": val_loss},
                )
```

In `main()`, add the CLI flag next to `--checkpoint-interval`:

```python
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
```

And thread it into the `TrainConfig(...)` construction, next to `checkpoint_interval=args.checkpoint_interval,`:

```python
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — every test in the project, including all of `tests/test_streaming.py` and `tests/test_train_helpers.py` from earlier tasks.

Also run: `uv run mypy src/` and `uv run ruff check .` — both must be clean (aside from any pre-existing unrelated finding that predates this branch).

- [ ] **Step 7: Manual smoke run to confirm the loop actually executes end to end**

Run:
```bash
uv run python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps 4 \
  --gradient-accumulation-steps 2 --batch-size 2 --checkpoint-interval 2 --eval-interval 2 \
  --wandb-mode disabled
```
Expected: completes without error; the JSONL log (`app.log`) contains `val_loss` entries at steps 2 and 4 alongside the existing `step %d complete` and `saved checkpoint` entries; `checkpoints/step_2.pt` and `checkpoints/step_4.pt` are written.

- [ ] **Step 8: Commit**

```bash
git add src/llmtrain/data/streaming.py src/llmtrain/training/train.py tests/test_streaming.py
git commit -m "feat: wire held-out validation loss into the training loop"
```

---

## Self-Review Notes

- **Spec coverage:** All four `## Components` sections of `2026-08-06-validation-loop-design.md` map to a task — `DatasetSpec`/registry → Task 1, config addition → Task 2, `evaluate()` → Task 3, `load_streaming_datasets` + loop integration → Task 4. The spec's "Explicitly deferred" items (best-checkpoint selection, per-run CLI override of holdout size) required no task by design.
- **Type consistency:** `load_streaming_datasets(dataset_name: str, seed: int, buffer_size: int, load_fn=load_dataset) -> tuple[IterableDataset, IterableDataset]` is defined once (Task 4) and consumed once, at its only call site in the same task — no drift possible. `evaluate(model, val_dataloader, pad_id, device, autocast_dtype, use_amp) -> float` is defined in Task 3 and called with the same positional order in Task 4.
- **Sequencing note:** Tasks 1–3 are deliberately non-breaking and independent (any order works); Task 4 is the only task where a breaking rename and its fix are inseparable, so it does both in one task/commit rather than splitting further — consistent with how the prior `pretraining-loop-hardening` plan handled its own load-bearing loop restructuring as a single task.
- **Placeholder scan:** no TBD/TODO; every step includes literal code, not a description of code.
