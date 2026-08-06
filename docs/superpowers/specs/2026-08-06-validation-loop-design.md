# Held-Out Validation Loop — Design

Date: 2026-08-06

## Scope

Second of three specs from this session's brainstorming (after `2026-08-06-pretraining-loop-hardening-design.md`; fused cross-entropy is the remaining one). Closes the "no validation signal at all" gap flagged by this session's earlier audit — currently `train()` only ever logs train loss, so there's no way to notice overfitting or judge generalization during a long paid A100 run.

**In scope:** `DatasetSpec`/`DATASET_REGISTRY` changes to declare a validation strategy per dataset, `load_streaming_dataset` → `load_streaming_datasets` (renamed, returns a train/val pair), a new `evaluate()` helper, and its integration into `train()`'s loop.

**Explicitly deferred / excluded:**

- Best-checkpoint selection / early stopping based on val loss — this spec only adds the loss _signal_, not a model-selection system.
- Sequence packing, `num_workers` tuning, fused cross-entropy, MoE, gradient checkpointing — separate specs or explicitly rejected (see prior specs and the architecture-modernization audit this session started from).
- Per-run CLI override of held-out size — moved into `DatasetSpec` instead (see Components §1); rationale below.

## Investigation: what validation data actually exists

Checked directly against the Hugging Face Hub (`datasets.get_dataset_split_names`) rather than assuming:

| Dataset                                      | Splits available |
| -------------------------------------------- | ---------------- |
| `Trelis/tiny-shakespeare`                    | `train`, `test`  |
| `reds0510/enwik8-processed`                  | `train` only     |
| `HuggingFaceFW/fineweb-edu` (`sample-100BT`) | `train` only     |

Only `tiny_shakespeare` has a real held-out split. The other two — including the dataset the real pretraining run actually uses — need a validation set carved out of `train`. A uniform "just point at a `validation` split" design doesn't work for 2 of the 3 registered datasets.

## Verification against current `datasets` library docs

Checked via context7 against the Hugging Face `datasets` docs, since this project already has one documented streaming/resume subtlety (`tiny_shakespeare` + shuffle buffer + `--resume`, per CLAUDE.md) and this design touches the same machinery:

- The documented pattern for splitting a streaming dataset is **`shuffle()` first, then `take(n)`/`skip(n)`** — not the reverse. `take`/`skip` "lock in shard order" and prevent correct shuffling afterward, so carving from an _unshuffled_ stream and shuffling the remainder separately (my first draft) would have been wrong. Canonical form: `shuffled = dataset.shuffle(seed, buffer_size); val = shuffled.take(n); train = shuffled.skip(n)`.
- The docs do **not** explicitly confirm or rule out whether a `skip()`-wrapped `IterableDataset` still round-trips correctly through `state_dict()`/`load_state_dict()` for exact `--resume`. Given the ambiguity and the project's existing caution around this exact interaction, this needs an explicit unit test (§ Testing) rather than an assumption — the same standard already applied to the known `tiny_shakespeare` resume bug.

## Components

### 1. `DatasetSpec` + registry: validation strategy lives per-dataset

Initial draft put a single `DataConfig.val_holdout_examples: int` global field on the training run config. Rejected during design review: that field would be silently ignored for any dataset with a real split (`tiny_shakespeare`), with nothing in the config surface indicating which datasets it actually applies to — exactly the kind of "shittily implemented" gap this session set out to find. "How to get validation data for this dataset" is a property of the dataset, the same reasoning that already puts `path`/`text_column`/`split` in `DatasetSpec` rather than as CLI-tunable globals.

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
```

```python
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tiny_shakespeare": DatasetSpec(
        path="Trelis/tiny-shakespeare", name=None, split="train",
        text_column="Text", val_split="test",
    ),
    "reformer_enwik8": DatasetSpec(
        path="reds0510/enwik8-processed", name=None, split="train",
        val_holdout_examples=1000,
    ),
    "fineweb_edu": DatasetSpec(
        path="HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train",
        val_holdout_examples=1000,
    ),
}
```

Every entry is forced to make an explicit, mutually exclusive choice — `__post_init__` runs at module-import time (when the `DATASET_REGISTRY` dict literal is evaluated), so a malformed future entry (both fields set, or neither) fails fast at import/test-collection time, before it ever reaches a training run. `DataConfig` gains no new field for this; `val_holdout_examples` is not CLI-tunable, matching the existing treatment of the rest of `DatasetSpec`.

### 2. `load_streaming_dataset` → `load_streaming_datasets` (renamed, returns a pair)

```python
def load_streaming_datasets(
    dataset_name: str,
    seed: int,
    buffer_size: int,
    load_fn: Callable[..., IterableDataset] = load_dataset,
) -> tuple[IterableDataset, IterableDataset]:  # (train, val)
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

- Native-split path: val is loaded independently from a genuinely disjoint Hub split — no interaction with train's shuffle buffer or resume state at all.
- Carve path: shuffle-then-split per the verified `datasets` pattern above; train and val are disjoint by construction since `skip`/`take` partition the same shuffled stream at the same cut point.
- `train.py`'s one call site updates to unpack both and build a second `DataLoader` for val (`pin_memory=True`, `drop_last=True`, same `collate_fn` as train — `drop_last` matters here too, so eval forward passes don't force a `torch.compile` recompilation on a ragged final batch).

### 3. Config addition

`TrainConfig.eval_interval: int = 500` (+ `--eval-interval` CLI flag). Consulted in optimizer-step terms (per the step-semantics redefinition in the pretraining-loop-hardening spec) alongside the existing `checkpoint_interval` check.

### 4. `evaluate()` + loop integration

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

Called every `eval_interval` optimizer steps inside `train()`'s loop (same place the `checkpoint_interval` check lives); result logged to W&B as `val_loss` at the current step. The val dataloader is small/finite by construction (bounded by `val_holdout_examples`, or the size of the native `test` split), so each call cheaply re-iterates it from scratch — no state to persist, no interaction with `--resume` needed on the val side.

## Error handling

`DatasetSpec.__post_init__` raises `ValueError` on a malformed registry entry (both `val_split` and `val_holdout_examples` set, or neither) — fails at import time, matching the existing validation style in `transformer.py`. No other new error paths; `eval_interval` is a trusted config value, consistent with the project's existing convention of not validating internal/trusted config.

## Testing strategy

CPU-only, tiny fake data, per CLAUDE.md's existing testing strategy — extending `tests/test_streaming.py`:

- `DatasetSpec.__post_init__`: constructing an entry with both `val_split` and `val_holdout_examples` set (or neither) raises `ValueError`.
- Registry sanity: `tiny_shakespeare` has `val_split="test"`, `val_holdout_examples=None`; the other two have `val_split=None`, `val_holdout_examples=1000`.
- Carve path: fake `load_fn` returning 20 examples, a `DatasetSpec` with `val_holdout_examples=5` → val has exactly 5 examples, train has the disjoint remaining 15 (assert on content, not just count, to catch an off-by-one or overlap bug).
- Native-split path: fake `load_fn` that returns different content depending on the `split` argument passed in → assert the val dataset's content matches what was requested via `spec.val_split`, not the train split.
- **Resume interaction (load-bearing, addresses the docs ambiguity above)**: small in-memory `IterableDataset`, apply `shuffle().skip(n)`, iterate partway, save `state_dict()`, construct a fresh `skip(n)`-wrapped dataset from the same source, `load_state_dict()`, and confirm iteration resumes at the correct position with no dropped or duplicated examples.
- `evaluate()`: tiny model + tiny fake val `DataLoader`, assert it returns a finite float and that `model.training` is restored to its original value afterward.
- `train()`/`main()` orchestration itself remains untested by design (per existing convention) — validated by manual smoke test instead, same open follow-up noted in the pretraining-loop-hardening spec (`docs/smoke-test.md` was deleted this session).

## Config changes summary

`DatasetSpec` gains `val_split: str | None = None`, `val_holdout_examples: int | None = None`, and a `__post_init__` mutual-exclusivity check. `TrainConfig` gains `eval_interval: int = 500`. `load_streaming_dataset` renamed to `load_streaming_datasets`; return type changes from `IterableDataset` to `tuple[IterableDataset, IterableDataset]` — breaking change to the one call site in `train.py`, updated as part of this spec. No `DataConfig` field changes.
