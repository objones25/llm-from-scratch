# SFT Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add everything needed to run the SFT (supervised fine-tuning) stage on `smoltalk`/`no_robots` from a pretrained `fineweb_edu` checkpoint: chat-formatted data with prompt/response loss masking, a weights-only checkpoint init path distinct from `--resume`, and a `generate.py` stop-on-`[PAD]` change — per `docs/superpowers/specs/2026-08-08-sft-support-design.md`.

**Architecture:** A new `data/chat.py` module turns `messages: list[{role, content}]` into `(input_ids, labels)` pairs via ChatML-style tags through the existing frozen tokenizer, reusing `[PAD]` as both filler and end-of-turn stop signal (no vocab change). `training/train.py`'s loss functions and collate function move from a `pad_id`-masks-everything scheme to an explicit `labels` tensor (`IGNORE_INDEX = -100`) that's constructed once at collate time — this one interface serves both pretraining (mask padding) and SFT (mask prompt tokens) uniformly. A new `--init-from-checkpoint` flag loads pretrained weights only (fresh step counter, fresh dataset stream, tokenizer loaded from disk instead of retrained), alongside the existing `--resume`.

**Tech Stack:** PyTorch, Hugging Face `tokenizers`/`datasets`, existing `llmtrain` package conventions (dataclass configs, CPU-only unit tests with tiny fake data).

## Global Constraints

- CPU-only, tiny fake data for every automated test — no GPU, no network, no cost (per CLAUDE.md's Testing strategy).
- `train()`/`main()` orchestration stays untested by design; validated by a manual smoke test only (per CLAUDE.md and the design spec's Testing strategy).
- No new `TrainConfig`/`ModelConfig`/`GenerationConfig`/`DataConfig` dataclass fields — this spec is scoped to CLI flags and one new `DatasetSpec` field (`messages_column`).
- No new special vocab tokens or embedding resizing — `[PAD]` is reused as the stop signal.
- `ruff check .`, `uv run mypy src/`, and `uv run pytest` must stay clean after every task.
- Commit after each task (not each step) unless a step says otherwise.

---

## Task 1: `data/chat.py` — chat formatting and loss masking

**Files:**
- Create: `src/llmtrain/data/chat.py`
- Test: Create `tests/test_chat.py`

**Interfaces:**
- Produces: `IGNORE_INDEX: int = -100`, `format_turn(role: str, content: str) -> str`, `encode_chat_example(tokenizer: Tokenizer, messages: list[dict], pad_id: int, max_seq_len: int) -> tuple[list[int], list[int]]`, `encode_chat_batch(tokenizer: Tokenizer, examples: list[dict], pad_id: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]`. Task 3 imports `IGNORE_INDEX` and `encode_chat_batch` from this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat.py`:

```python
import torch

from llmtrain.data.chat import IGNORE_INDEX, encode_chat_batch, encode_chat_example, format_turn
from llmtrain.data.tokenizer import PAD_TOKEN, train_tokenizer


def _tiny_tokenizer():
    texts = [
        format_turn("system", "be nice"),
        format_turn("user", "hi there"),
        format_turn("assistant", "hello"),
    ]
    return train_tokenizer(texts, vocab_size=200)


def test_format_turn_wraps_content_in_role_tags():
    assert format_turn("user", "hi") == "<|user|>\nhi\n"


def test_encode_chat_example_masks_non_assistant_turns_and_supervises_assistant_turns():
    tokenizer = _tiny_tokenizer()
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello"},
    ]
    max_seq_len = 64

    input_ids, labels = encode_chat_example(tokenizer, messages, pad_id, max_seq_len)

    assert len(input_ids) == max_seq_len
    assert len(labels) == max_seq_len

    system_ids = tokenizer.encode(format_turn("system", "be nice")).ids
    user_ids = tokenizer.encode(format_turn("user", "hi there")).ids
    assistant_ids = tokenizer.encode(format_turn("assistant", "hello")).ids

    non_assistant_len = len(system_ids) + len(user_ids)
    assert labels[:non_assistant_len] == [IGNORE_INDEX] * non_assistant_len
    assert input_ids[:non_assistant_len] == system_ids + user_ids

    assistant_start = non_assistant_len
    assistant_end = assistant_start + len(assistant_ids)
    assert labels[assistant_start:assistant_end] == assistant_ids
    assert input_ids[assistant_start:assistant_end] == assistant_ids

    # exactly one pad_id spliced in right after the assistant turn, and supervised
    assert input_ids[assistant_end] == pad_id
    assert labels[assistant_end] == pad_id

    # tail padding beyond that is never supervised
    assert all(label == IGNORE_INDEX for label in labels[assistant_end + 1 :])
    assert all(token_id == pad_id for token_id in input_ids[assistant_end + 1 :])


def test_encode_chat_example_truncates_to_exactly_max_seq_len():
    tokenizer = _tiny_tokenizer()
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    messages = [
        {"role": "user", "content": "hi there, how are you doing today my friend"},
        {"role": "assistant", "content": "I am doing great thanks for asking, how about you"},
    ]
    max_seq_len = 5

    input_ids, labels = encode_chat_example(tokenizer, messages, pad_id, max_seq_len)

    assert len(input_ids) == max_seq_len
    assert len(labels) == max_seq_len


def test_encode_chat_batch_stacks_examples_into_tensors():
    tokenizer = _tiny_tokenizer()
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    examples = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {
            "messages": [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "hi"},
            ]
        },
    ]

    input_ids, labels = encode_chat_batch(tokenizer, examples, pad_id, max_seq_len=16)

    assert input_ids.shape == (2, 16)
    assert labels.shape == (2, 16)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.data.chat'`

- [ ] **Step 3: Write the implementation**

Create `src/llmtrain/data/chat.py`:

```python
import torch
from tokenizers import Tokenizer

IGNORE_INDEX = -100  # shared with training/train.py's loss functions


def format_turn(role: str, content: str) -> str:
    return f"<|{role}|>\n{content}\n"


def encode_chat_example(
    tokenizer: Tokenizer, messages: list[dict], pad_id: int, max_seq_len: int
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        turn_ids = tokenizer.encode(format_turn(message["role"], message["content"])).ids
        input_ids.extend(turn_ids)
        if message["role"] == "assistant":
            labels.extend(turn_ids)  # supervised
            input_ids.append(pad_id)  # stop signal
            labels.append(pad_id)  # ...and it's supervised too
        else:
            labels.extend([IGNORE_INDEX] * len(turn_ids))
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    pad_amount = max_seq_len - len(input_ids)
    input_ids.extend([pad_id] * pad_amount)
    labels.extend([IGNORE_INDEX] * pad_amount)  # tail filler: never supervised
    return input_ids, labels


def encode_chat_batch(
    tokenizer: Tokenizer, examples: list[dict], pad_id: int, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [
        encode_chat_example(tokenizer, ex["messages"], pad_id, max_seq_len) for ex in examples
    ]
    input_ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    labels = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return input_ids, labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add src/llmtrain/data/chat.py tests/test_chat.py
git commit -m "$(cat <<'EOF'
Add data/chat.py: chat-formatted encoding with assistant-turn loss masking

Turn-by-turn tokenize-and-concatenate through the existing frozen
byte-level BPE tokenizer, with [PAD] reused as an end-of-turn stop signal
(supervised, unlike padding filler) rather than adding a new vocab token.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `data/streaming.py` — chat dataset registry entries

**Files:**
- Modify: `src/llmtrain/data/streaming.py`
- Test: Modify `tests/test_streaming.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `DatasetSpec.messages_column: str | None = None`; `DATASET_REGISTRY["smoltalk"]`, `DATASET_REGISTRY["no_robots"]`. Task 3/4 read `DATASET_REGISTRY[data_cfg.dataset_name].messages_column`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_streaming.py` (after the existing `test_reformer_enwik8_and_fineweb_edu_registry_entries_carve_val_holdout` test):

```python
def test_dataset_spec_defaults_messages_column_to_none():
    spec = DatasetSpec(path="x", name=None, split="train", val_split="test")
    assert spec.messages_column is None


def test_smoltalk_registry_entry_uses_messages_column_and_native_val_split():
    spec = DATASET_REGISTRY["smoltalk"]
    assert spec.path == "HuggingFaceTB/smoltalk"
    assert spec.name == "all"
    assert spec.messages_column == "messages"
    assert spec.val_split == "test"
    assert spec.val_holdout_examples is None


def test_no_robots_registry_entry_uses_messages_column_and_native_val_split():
    spec = DATASET_REGISTRY["no_robots"]
    assert spec.path == "HuggingFaceH4/no_robots"
    assert spec.name == "default"
    assert spec.messages_column == "messages"
    assert spec.val_split == "test"
    assert spec.val_holdout_examples is None


def test_load_streaming_datasets_skips_rename_when_messages_column_is_set(monkeypatch):
    def _fake_chat_load_dataset(path, name, split, streaming):
        return Dataset.from_dict(
            {"messages": [[{"role": "user", "content": f"hi {i}"}] for i in range(10)]}
        ).to_iterable_dataset(num_shards=1)

    monkeypatch.setitem(
        DATASET_REGISTRY,
        "chat_test",
        DatasetSpec(
            path="x",
            name=None,
            split="train",
            text_column="Text",  # would normally trigger rename_column("Text", "text")
            messages_column="messages",
            val_split="train",
        ),
    )
    train_dataset, _val_dataset = load_streaming_datasets(
        "chat_test", seed=42, buffer_size=5, load_fn=_fake_chat_load_dataset
    )

    examples = list(train_dataset)
    assert all("messages" in example for example in examples)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming.py -v -k "messages_column or smoltalk or no_robots"`
Expected: FAIL — `TypeError: DatasetSpec.__init__() got an unexpected keyword argument 'messages_column'` (for the first three), and a `KeyError`/lookup failure for the registry-entry tests since `smoltalk`/`no_robots` don't exist yet.

- [ ] **Step 3: Write the implementation**

In `src/llmtrain/data/streaming.py`, update the `DatasetSpec` dataclass:

```python
@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    split: str
    text_column: str = "text"
    val_split: str | None = None
    val_holdout_examples: int | None = None
    messages_column: str | None = None

    def __post_init__(self) -> None:
        if (self.val_split is None) == (self.val_holdout_examples is None):
            raise ValueError("exactly one of val_split or val_holdout_examples must be set")
```

Add two entries to `DATASET_REGISTRY` (after `fineweb_edu`):

```python
    "smoltalk": DatasetSpec(
        path="HuggingFaceTB/smoltalk",
        name="all",
        split="train",
        messages_column="messages",
        val_split="test",
    ),
    "no_robots": DatasetSpec(
        path="HuggingFaceH4/no_robots",
        name="default",
        split="train",
        messages_column="messages",
        val_split="test",
    ),
```

Update `load_streaming_datasets`'s rename guards (there are two identical `if spec.text_column != "text":` checks — one for `dataset`, one for `val_dataset`) to also require `spec.messages_column is None`:

```python
    dataset = load_fn(spec.path, name=spec.name, split=spec.split, streaming=True)
    if spec.text_column != "text" and spec.messages_column is None:
        dataset = dataset.rename_column(spec.text_column, "text")
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    if spec.val_split is not None:
        val_dataset = load_fn(spec.path, name=spec.name, split=spec.val_split, streaming=True)
        if spec.text_column != "text" and spec.messages_column is None:
            val_dataset = val_dataset.rename_column(spec.text_column, "text")
        return shuffled, list(val_dataset)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: PASS (all tests, including pre-existing ones)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add src/llmtrain/data/streaming.py tests/test_streaming.py
git commit -m "$(cat <<'EOF'
Add smoltalk/no_robots to the dataset registry for SFT

DatasetSpec gains messages_column: str | None, set for both new entries.
When set, load_streaming_datasets skips the text_column rename entirely —
chat examples keep their messages column (and everything else) untouched.
Both datasets already expose a native train/test split, so no holdout-
carving path is needed, unlike reformer_enwik8/fineweb_edu.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `training/train.py` — unified `(input_ids, labels)` loss interface

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Test: Modify `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `IGNORE_INDEX`, `encode_chat_batch` from `llmtrain.data.chat` (Task 1); `DATASET_REGISTRY` from `llmtrain.data.streaming` (Task 2, field `messages_column`).
- Produces: `next_token_loss(logits, labels) -> Tensor`, `next_token_loss_fused(hidden, head_weight, labels) -> Tensor`, `compute_loss(model, input_ids, labels, use_fused_ce) -> Tensor`, `evaluate(model, val_dataloader, device, autocast_dtype, use_amp, use_fused_ce) -> float`, `make_collate_fn(tokenizer, max_seq_len, messages_column) -> Callable[[list[dict]], tuple[Tensor, Tensor]]`, `train_step(model, optimizer, batches: list[tuple[Tensor, Tensor]], train_cfg, device, autocast_dtype, use_fused_ce, step) -> tuple[float, float, float]`, `collect_micro_batches` generic over batch type via `TypeVar`. All drop `pad_id` as an explicit parameter — masking now lives in `labels`. Task 4 builds directly on this `train()` body.

- [ ] **Step 1: Write the failing tests**

In `tests/test_train_helpers.py`, replace the whole file's contents with:

```python
import argparse
import math

import pytest
import torch
from datasets import Dataset

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig
from llmtrain.training.train import (
    build_configs_from_args,
    collect_micro_batches,
    compute_loss,
    evaluate,
    get_lr,
    load_or_train_tokenizer,
    make_collate_fn,
    next_token_loss,
    select_device,
    train_step,
)


def _fake_train_dataset(texts: list[str]):
    return Dataset.from_dict({"text": texts}).to_iterable_dataset(num_shards=1)


def test_select_device_returns_a_torch_device():
    device = select_device()
    assert isinstance(device, torch.device)


def test_next_token_loss_is_near_zero_for_perfect_predictions():
    vocab_size = 4
    labels = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, vocab_size), -100.0)
    for position, target_id in enumerate(labels[0, 1:]):
        logits[0, position, target_id] = 100.0
    loss = next_token_loss(logits, labels)
    assert loss.item() < 0.01


def test_next_token_loss_ignores_ignore_index_positions():
    vocab_size = 4
    logits = torch.randn(1, 3, vocab_size)
    labels_with_ignored = torch.tensor([[0, -100, 2]])
    labels_all_real = torch.tensor([[0, 1, 2]])
    # Wrong prediction at the ignored position shouldn't move the loss at all.
    loss_ignored = next_token_loss(logits, labels_with_ignored)
    loss_real = next_token_loss(logits, labels_all_real)
    assert loss_ignored.item() != pytest.approx(loss_real.item())


def test_make_collate_fn_encodes_a_batch_of_pretraining_examples():
    texts = ["hello world", "hello there", "the quick brown fox"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    collate = make_collate_fn(tokenizer, max_seq_len=5, messages_column=None)

    input_ids, labels = collate([{"text": "hello world"}, {"text": "hello there"}])

    assert input_ids.shape == (2, 5)
    assert input_ids.dtype == torch.long
    assert labels.shape == (2, 5)
    pad_id = tokenizer.token_to_id("[PAD]")
    assert torch.equal(labels == -100, input_ids == pad_id)


def test_make_collate_fn_encodes_a_batch_of_chat_examples():
    texts = ["<|user|>\nhi\n", "<|assistant|>\nhello\n"]
    tokenizer = train_tokenizer(texts, vocab_size=100)
    collate = make_collate_fn(tokenizer, max_seq_len=20, messages_column="messages")
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]

    input_ids, labels = collate(examples)

    assert input_ids.shape == (1, 20)
    assert labels.shape == (1, 20)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_get_lr_ramps_linearly_during_warmup():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(0, cfg) == pytest.approx(0.1)
    assert get_lr(9, cfg) == pytest.approx(1.0)


def test_get_lr_decays_smoothly_from_end_of_warmup():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(10, cfg) == pytest.approx(1.0)


def test_get_lr_decreases_monotonically_during_decay():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(50, cfg) > get_lr(90, cfg)


def test_get_lr_clamps_to_min_lr_after_max_steps():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(100, cfg) == pytest.approx(0.1)
    assert get_lr(150, cfg) == pytest.approx(0.1)


def test_get_lr_with_zero_warmup_starts_at_max_lr():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=0, max_steps=100)
    assert get_lr(0, cfg) == pytest.approx(1.0)


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


def test_evaluate_returns_finite_float_and_restores_train_mode():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.train()

    input_ids = torch.randint(0, 16, (2, 6))
    labels = input_ids.clone()
    val_dataloader = [(input_ids, labels), (input_ids, labels)]

    val_loss = evaluate(
        model,
        val_dataloader,
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

    input_ids = torch.randint(0, 16, (2, 6))
    labels = input_ids.clone()
    val_dataloader = [(input_ids, labels)]

    evaluate(
        model,
        val_dataloader,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
        use_fused_ce=False,
    )

    assert model.training is False


def test_compute_loss_non_fused_matches_direct_next_token_loss():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, 16, (2, 6))
    labels = input_ids.clone()

    with torch.no_grad():
        loss_via_compute_loss = compute_loss(model, input_ids, labels, use_fused_ce=False)
        logits = model(input_ids)
        loss_direct = next_token_loss(logits, labels)

    assert torch.allclose(loss_via_compute_loss, loss_direct, atol=1e-6)


def test_collect_micro_batches_collects_n_batches_in_order():
    dataloader = [torch.tensor([i]) for i in range(5)]
    data_iter = iter(dataloader)

    batches, data_iter = collect_micro_batches(dataloader, data_iter, 3)

    assert [b.item() for b in batches] == [0, 1, 2]


def test_collect_micro_batches_wraps_to_a_new_epoch_on_exhaustion():
    dataloader = [torch.tensor([i]) for i in range(2)]
    data_iter = iter(dataloader)

    batches, data_iter = collect_micro_batches(dataloader, data_iter, 3)

    assert [b.item() for b in batches] == [0, 1, 0]
    assert next(data_iter).item() == 1


def test_train_step_updates_parameters_and_zeros_grads_after():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    train_cfg = TrainConfig(grad_clip=1.0, lr=1e-2, min_lr=1e-3, warmup_steps=0, max_steps=100)
    batch_1 = torch.randint(0, 16, (2, 6))
    batch_2 = torch.randint(0, 16, (2, 6))
    batches = [(batch_1, batch_1.clone()), (batch_2, batch_2.clone())]
    params_before = [p.clone() for p in model.parameters()]

    avg_loss, grad_norm, lr = train_step(
        model,
        optimizer,
        batches,
        train_cfg,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_fused_ce=False,
        step=0,
    )

    assert math.isfinite(avg_loss)
    assert grad_norm >= 0
    assert lr == pytest.approx(get_lr(0, train_cfg))
    assert any(
        not torch.equal(before, after) for before, after in zip(params_before, model.parameters())
    )
    assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters())


def test_build_configs_from_args_maps_every_field_to_its_dataclass():
    args = argparse.Namespace(
        dataset="tiny_shakespeare",
        shuffle_buffer_size=123,
        max_seq_len=64,
        tokenizer_vocab_size=500,
        tokenizer_sample_size=50,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        dropout=0.1,
        rope_theta=5000.0,
        batch_size=4,
        gradient_accumulation_steps=2,
        grad_clip=0.5,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=10,
        weight_decay=0.05,
        beta1=0.8,
        beta2=0.9,
        max_steps=100,
        seed=7,
        checkpoint_dir="ckpt",
        checkpoint_interval=10,
        keep_last_n_checkpoints=2,
        eval_interval=20,
        compile=False,
        use_amp=False,
        use_fused_ce=False,
        wandb_project="proj",
        wandb_mode="disabled",
        log_file="test.log",
        resume=None,
    )

    data_cfg, model_cfg, train_cfg = build_configs_from_args(args)

    assert data_cfg == DataConfig(
        dataset_name="tiny_shakespeare",
        shuffle_buffer_size=123,
        max_seq_len=64,
        tokenizer_vocab_size=500,
        tokenizer_sample_size=50,
    )
    assert model_cfg == ModelConfig(
        d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.1, rope_theta=5000.0
    )
    assert train_cfg == TrainConfig(
        batch_size=4,
        gradient_accumulation_steps=2,
        grad_clip=0.5,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=10,
        weight_decay=0.05,
        beta1=0.8,
        beta2=0.9,
        max_steps=100,
        seed=7,
        checkpoint_dir="ckpt",
        checkpoint_interval=10,
        keep_last_n_checkpoints=2,
        eval_interval=20,
        compile=False,
        use_amp=False,
        use_fused_ce=False,
        wandb_project="proj",
        wandb_mode="disabled",
        log_file="test.log",
    )


def test_load_or_train_tokenizer_trains_fresh_when_not_resuming():
    train_dataset = _fake_train_dataset(["hello world", "hello there", "the quick brown fox"])
    data_cfg = DataConfig(tokenizer_vocab_size=50, tokenizer_sample_size=3)

    tokenizer = load_or_train_tokenizer(None, train_dataset, data_cfg)

    assert tokenizer.get_vocab_size() > 0


def test_load_or_train_tokenizer_loads_saved_tokenizer_when_resuming(tmp_path):
    texts = ["hello world", "hello there"]
    train_dataset = _fake_train_dataset(texts)
    data_cfg = DataConfig(tokenizer_vocab_size=50, tokenizer_sample_size=2)
    saved_tokenizer = train_tokenizer(texts, vocab_size=50)
    saved_tokenizer.save(str(tmp_path / "tokenizer.json"))
    resume_path = str(tmp_path / "step_10.pt")

    loaded = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)

    assert loaded.get_vocab() == saved_tokenizer.get_vocab()


def test_load_or_train_tokenizer_falls_back_when_resuming_without_a_saved_tokenizer(tmp_path):
    texts = ["hello world", "hello there", "the quick brown fox"]
    train_dataset = _fake_train_dataset(texts)
    data_cfg = DataConfig(tokenizer_vocab_size=50, tokenizer_sample_size=3)
    resume_path = str(tmp_path / "step_10.pt")  # no tokenizer.json alongside it

    tokenizer = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)

    assert tokenizer.get_vocab_size() > 0
```

(This drops the old `next_token_loss_is_near_zero...`/`make_collate_fn_encodes_a_batch_of_examples`/`evaluate`/`compute_loss`/`train_step` tests' old `pad_id`-based signatures and replaces them with the `labels`-based versions above, and adds the two new `make_collate_fn`/`next_token_loss` tests.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: FAIL — `TypeError` on the new/updated call sites (e.g. `next_token_loss() missing 1 required positional argument`, `make_collate_fn() missing 1 required positional argument: 'messages_column'`, `evaluate() got an unexpected keyword argument`).

- [ ] **Step 3: Write the implementation**

In `src/llmtrain/training/train.py`, update the imports block at the top of the file to:

```python
import argparse
import logging
import math
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import torch
from datasets import IterableDataset
from tokenizers import Tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from llmtrain.data.chat import IGNORE_INDEX, encode_chat_batch
from llmtrain.data.streaming import DATASET_REGISTRY, load_streaming_datasets
from llmtrain.data.tokenizer import PAD_TOKEN, encode_batch, train_tokenizer
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.checkpoint import load_checkpoint, prune_old_checkpoints, save_checkpoint
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig

logger = logging.getLogger(__name__)

_Batch = TypeVar("_Batch")
```

Replace `next_token_loss` through `make_collate_fn` (everything from `def next_token_loss(...)` down to the end of `def make_collate_fn(...)`) with:

```python
def next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def next_token_loss_fused(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    from liger_kernel.transformers import (  # type: ignore[import-not-found]
        LigerFusedLinearCrossEntropyLoss,
    )

    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return loss_fn(head_weight, shift_hidden, shift_labels)


def compute_loss(
    model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, use_fused_ce: bool
) -> torch.Tensor:
    if use_fused_ce:
        hidden = model(input_ids, return_hidden=True)  # type: ignore[misc]
        head_weight = model.token_emb.weight  # type: ignore[union-attr,attr-defined]
        return next_token_loss_fused(hidden, head_weight, labels)  # type: ignore[arg-type]
    logits = model(input_ids)
    return next_token_loss(logits, labels)


def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    use_amp: bool,
    use_fused_ce: bool,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for input_ids, labels in val_dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                losses.append(compute_loss(model, input_ids, labels, use_fused_ce).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def make_collate_fn(
    tokenizer: Tokenizer, max_seq_len: int, messages_column: str | None
) -> Callable[[list[dict]], tuple[torch.Tensor, torch.Tensor]]:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)

    def collate(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        if messages_column is not None:
            return encode_chat_batch(tokenizer, examples, pad_id, max_seq_len)
        input_ids = encode_batch(tokenizer, [ex["text"] for ex in examples], max_seq_len)
        labels = input_ids.clone()
        labels[input_ids == pad_id] = IGNORE_INDEX
        return input_ids, labels

    return collate
```

Replace `collect_micro_batches` with the genericized version:

```python
def collect_micro_batches(
    dataloader: Iterable[_Batch], data_iter: Iterator[_Batch], n: int
) -> tuple[list[_Batch], Iterator[_Batch]]:
    # A dataloader smaller than n (e.g. tiny_shakespeare with a large
    # --gradient-accumulation-steps) can wrap around more than once per call — each
    # StopIteration starts a fresh epoch rather than raising past a bare `next()`.
    batches: list[_Batch] = []
    for _ in range(n):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batches.append(batch)
    return batches, data_iter
```

Replace `train_step` with:

```python
def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_cfg: TrainConfig,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    use_fused_ce: bool,
    step: int,
) -> tuple[float, float, float]:
    accumulated_loss = 0.0
    for input_ids, labels in batches:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=train_cfg.use_amp):
            loss = compute_loss(model, input_ids, labels, use_fused_ce) / len(batches)
        loss.backward()
        accumulated_loss += loss.item()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

    lr = get_lr(step, train_cfg)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    optimizer.zero_grad()

    return accumulated_loss, total_norm.item(), lr
```

Finally, replace the whole `train()` function body with:

```python
def train(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    resume_path: str | None = None,
) -> None:
    configure_logging(log_file=train_cfg.log_file)
    # Must be set before any CUDA allocation happens (the allocator reads it lazily on
    # first use) — reduces fragmentation-driven OOMs on long runs. No-op on MPS/CPU.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(train_cfg.seed)
    device = select_device()
    logger.info("training on device %s", device.type, extra={"device": device.type})
    if device.type == "cuda":
        # TF32 matmuls: near-free throughput on Ampere+ (A100) with negligible precision
        # loss for a model already training under bf16 autocast; no effect on MPS/CPU.
        torch.set_float32_matmul_precision("high")

    train_dataset, val_dataset = load_streaming_datasets(
        data_cfg.dataset_name,
        seed=train_cfg.seed,
        buffer_size=data_cfg.shuffle_buffer_size,
    )
    tokenizer = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    model: torch.nn.Module = TransformerLM(model_cfg).to(device)
    if device.type == "cuda" and train_cfg.compile:
        # model.compile() (in-place) is the current PyTorch guidance over the older
        # functional torch.compile(model) wrapping — no reassignment needed, and
        # unlike the functional form it never wraps the model in an OptimizedModule,
        # so no _orig_mod.-prefixed state_dict keys, no attribute-proxying edge
        # cases, and generate.py (which always loads into a fresh, uncompiled model)
        # can never be affected by whatever wrapping strategy training used.
        model.compile()
    # GPT-3/LLaMA/nanoGPT-style two-group AdamW: exclude 1-D parameters (RMSNorm gains —
    # the only 1-D params left now that every nn.Linear is bias=False) from weight decay.
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    step = 0
    if resume_path is not None:
        # Training reconstructs the model from the current model_cfg, not the checkpoint's
        # persisted config — the returned model_config is only used by generate.py, which
        # rebuilds the model from scratch at inference time.
        step, dataset_state, _resumed_model_config = load_checkpoint(resume_path, model, optimizer)
        if dataset_state is not None:
            train_dataset.load_state_dict(dataset_state)
        logger.info("resumed from checkpoint at step %d", step, extra={"step": step})

    messages_column = DATASET_REGISTRY[data_cfg.dataset_name].messages_column
    dataloader = DataLoader(
        train_dataset,  # type: ignore[arg-type]  # IterableDataset isn't in DataLoader's stub overloads, but is supported at runtime
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        # A ragged final batch would force torch.compile to recompile for the new shape,
        # spiking memory mid-run; dropping it keeps every batch's shape constant.
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len, messages_column),
    )
    # val_dataset is a plain list[dict] (materialized by load_streaming_datasets to avoid
    # sharing mutable streaming state with train_dataset) — a valid map-style dataset at
    # runtime (any Sequence with __getitem__/__len__ works), but list isn't a subtype of
    # the Dataset[T] the stub declares, hence the ignore.
    val_dataloader: DataLoader[dict] = DataLoader(
        val_dataset,  # type: ignore[arg-type]  # list[dict] is a valid map-style dataset at runtime but isn't Dataset[T]
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len, messages_column),
    )

    wandb.init(
        project=train_cfg.wandb_project,
        mode=train_cfg.wandb_mode,  # type: ignore[arg-type]
        config={**asdict(train_cfg), **asdict(model_cfg)},
    )

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Must save before the first encode_batch call: encode_batch mutates the tokenizer's
    # truncation/padding state via enable_truncation/enable_padding, and that mutated state
    # gets serialized into tokenizer.json. Saving later would silently persist the wrong
    # truncation/padding length for anything (e.g. generate.py) that loads this file.
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    logger.info("saved tokenizer to %s", checkpoint_dir / "tokenizer.json")

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    use_fused_ce_effective = train_cfg.use_fused_ce and device.type == "cuda"

    model.train()
    optimizer.zero_grad()
    # A bare `for batch in dataloader` would stop as soon as the underlying stream is
    # exhausted, capping training at whatever step count one pass through the dataset
    # happens to reach — silently ignoring the rest of --max-steps. collect_micro_batches
    # re-iterates the dataset (a fresh epoch) whenever that happens. No-op for datasets
    # large enough to never exhaust within a normal run (reformer_enwik8, fineweb_edu);
    # this is what lets small datasets like tiny_shakespeare train for more than one epoch.
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]] = iter(dataloader)
    while step < train_cfg.max_steps:
        batches, data_iter = collect_micro_batches(
            dataloader, data_iter, train_cfg.gradient_accumulation_steps
        )
        avg_loss, grad_norm, lr = train_step(
            model,
            optimizer,
            batches,
            train_cfg,
            device,
            autocast_dtype,
            use_fused_ce_effective,
            step,
        )

        wandb.log({"loss": avg_loss, "lr": lr, "grad_norm": grad_norm}, step=step)
        logger.debug("step %d complete", step, extra={"step": step})

        step += 1
        if step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{step}.pt",
                model,
                optimizer,
                step=step,
                dataset_state=train_dataset.state_dict(),
            )
            prune_old_checkpoints(checkpoint_dir, train_cfg.keep_last_n_checkpoints)
            logger.info("saved checkpoint at step %d", step, extra={"step": step})
        if step % train_cfg.eval_interval == 0:
            val_loss = evaluate(
                model,
                val_dataloader,
                device,
                autocast_dtype,
                train_cfg.use_amp,
                use_fused_ce_effective,
            )
            wandb.log({"val_loss": val_loss}, step=step)
            logger.info(
                "val_loss %.4f at step %d",
                val_loss,
                step,
                extra={"step": step, "val_loss": val_loss},
            )

    wandb.finish()
    logger.info("training complete after %d steps", step, extra={"step": step})
```

Note `load_or_train_tokenizer` and `get_lr` are unchanged — leave them exactly as they are between `make_collate_fn` and `collect_micro_batches`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: PASS (all tests, full suite — this task touches shared loss/collate functions used across the file)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src/`
Expected: both clean. If mypy complains about `data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]] = iter(dataloader)` being incompatible with `_BaseDataLoaderIter`, the explicit annotation on that line (already present above) is the fix — confirm it's there.

- [ ] **Step 6: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py
git commit -m "$(cat <<'EOF'
Unify train.py's loss interface around an explicit labels tensor

next_token_loss/next_token_loss_fused/compute_loss/evaluate/train_step now
take a labels tensor (masked with IGNORE_INDEX=-100) instead of a pad_id
kwarg — pretraining's collate_fn constructs labels by masking padding
positions (same effective behavior as the old ignore_index=pad_id), and
make_collate_fn gains a messages_column parameter that routes chat
datasets through data/chat.py's encode_chat_batch instead. One interface
now serves both pretraining and the upcoming SFT stage.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `training/train.py` — `--init-from-checkpoint` weights-only init

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Test: Modify `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `resolve_local_path`, `sibling_path` from `llmtrain.s3` (existing, used unchanged from `generate.py`'s pattern); `train()`'s body from Task 3.
- Produces: `find_model_config_overrides(model_cfg: ModelConfig, saved_model_config: dict) -> dict[str, tuple]`; `train()` gains `init_from_checkpoint: str | None = None` and `tokenizer_path: str | None = None` parameters; `main()` gains `--init-from-checkpoint`/`--tokenizer-path` CLI flags (mutually exclusive with `--resume`) and extends `--dataset` choices with `"smoltalk"`, `"no_robots"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train_helpers.py` (add `from dataclasses import asdict` to the top imports, and add `find_model_config_overrides` to the `from llmtrain.training.train import (...)` block):

```python
def test_find_model_config_overrides_returns_empty_when_configs_match():
    model_cfg = ModelConfig(d_model=64, n_layers=4)
    saved = asdict(model_cfg)

    assert find_model_config_overrides(model_cfg, saved) == {}


def test_find_model_config_overrides_reports_mismatched_fields_excluding_vocab_size():
    model_cfg = ModelConfig(d_model=64, n_layers=4, vocab_size=999)
    saved = asdict(ModelConfig(d_model=128, n_layers=4, vocab_size=32768))

    overrides = find_model_config_overrides(model_cfg, saved)

    assert overrides == {"d_model": (64, 128)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_helpers.py -v -k find_model_config_overrides`
Expected: FAIL with `ImportError: cannot import name 'find_model_config_overrides'`

- [ ] **Step 3: Write the implementation**

In `src/llmtrain/training/train.py`, add the S3 import to the existing import block (alongside the other `llmtrain.*` imports):

```python
from llmtrain.s3 import resolve_local_path, sibling_path
```

Add `find_model_config_overrides` right before `def train(`:

```python
def find_model_config_overrides(
    model_cfg: ModelConfig, saved_model_config: dict
) -> dict[str, tuple[object, object]]:
    return {
        field: (getattr(model_cfg, field), saved_model_config[field])
        for field in saved_model_config
        if field != "vocab_size" and getattr(model_cfg, field) != saved_model_config[field]
    }
```

Update `train()`'s signature:

```python
def train(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    resume_path: str | None = None,
    init_from_checkpoint: str | None = None,
    tokenizer_path: str | None = None,
) -> None:
```

Replace this block (the tokenizer-loading + `model_cfg.vocab_size` line, right after `load_streaming_datasets`):

```python
    tokenizer = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)
    model_cfg.vocab_size = tokenizer.get_vocab_size()
```

with:

```python
    init_checkpoint_path: Path | None = None
    if init_from_checkpoint is not None:
        # SFT always starts from pretrained weights with a fresh tokenizer loaded from
        # disk, never a freshly retrained one over smoltalk/no_robots text — the SFT run
        # must use the exact tokenizer the pretrained embeddings were trained with.
        init_checkpoint_path = resolve_local_path(init_from_checkpoint)
        tokenizer_uri = tokenizer_path or sibling_path(init_from_checkpoint, "tokenizer.json")
        tokenizer = Tokenizer.from_file(str(resolve_local_path(tokenizer_uri)))
        raw_checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
        saved_model_config = raw_checkpoint.get("model_config")
        if saved_model_config is not None:
            overrides = find_model_config_overrides(model_cfg, saved_model_config)
            if overrides:
                logger.warning(
                    "model architecture flags disagree with checkpoint %s; the "
                    "checkpoint's values win: %s",
                    init_from_checkpoint,
                    overrides,
                )
            model_cfg = ModelConfig(
                **{**saved_model_config, "vocab_size": tokenizer.get_vocab_size()}
            )
        else:
            model_cfg.vocab_size = tokenizer.get_vocab_size()
    else:
        tokenizer = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)
        model_cfg.vocab_size = tokenizer.get_vocab_size()
```

Replace the resume block:

```python
    step = 0
    if resume_path is not None:
        # Training reconstructs the model from the current model_cfg, not the checkpoint's
        # persisted config — the returned model_config is only used by generate.py, which
        # rebuilds the model from scratch at inference time.
        step, dataset_state, _resumed_model_config = load_checkpoint(resume_path, model, optimizer)
        if dataset_state is not None:
            train_dataset.load_state_dict(dataset_state)
        logger.info("resumed from checkpoint at step %d", step, extra={"step": step})
```

with:

```python
    step = 0
    if init_from_checkpoint is not None:
        assert init_checkpoint_path is not None
        load_checkpoint(init_checkpoint_path, model, optimizer=None)
        logger.info("initialized weights from checkpoint %s", init_from_checkpoint)
    elif resume_path is not None:
        # Training reconstructs the model from the current model_cfg, not the checkpoint's
        # persisted config — the returned model_config is only used by generate.py, which
        # rebuilds the model from scratch at inference time.
        step, dataset_state, _resumed_model_config = load_checkpoint(resume_path, model, optimizer)
        if dataset_state is not None:
            train_dataset.load_state_dict(dataset_state)
        logger.info("resumed from checkpoint at step %d", step, extra={"step": step})
```

In `main()`, extend the `--dataset` choices:

```python
    parser.add_argument(
        "--dataset",
        choices=["tiny_shakespeare", "reformer_enwik8", "fineweb_edu", "smoltalk", "no_robots"],
        default=DataConfig.dataset_name,
    )
```

Replace the single `parser.add_argument("--resume", type=str, default=None)` line with a mutually exclusive group plus the new `--tokenizer-path` flag:

```python
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=str, default=None)
    resume_group.add_argument("--init-from-checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
```

Finally, update the last two lines of `main()`:

```python
    data_cfg, model_cfg, train_cfg = build_configs_from_args(args)
    train(
        data_cfg,
        model_cfg,
        train_cfg,
        resume_path=args.resume,
        init_from_checkpoint=args.init_from_checkpoint,
        tokenizer_path=args.tokenizer_path,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: PASS (full suite)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src/`
Expected: both clean

- [ ] **Step 6: Verify the CLI wiring manually**

Run: `uv run python -m llmtrain.training.train --resume x --init-from-checkpoint y`
Expected: argparse error, `argument --init-from-checkpoint: not allowed with argument --resume` (confirms the mutually exclusive group works before any dataset/model loading happens)

Run: `uv run python -m llmtrain.training.train --dataset smoltalk --help`
Expected: no error (confirms `"smoltalk"` is now a valid `--dataset` choice)

- [ ] **Step 7: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py
git commit -m "$(cat <<'EOF'
Add --init-from-checkpoint weights-only init for the SFT stage

Loads pretrained model weights (optimizer=None) and the tokenizer from
disk (via s3.py's resolve_local_path/sibling_path, same helpers
generate.py already uses) instead of retraining one — fresh step counter,
fresh dataset stream over the SFT dataset. Model architecture is
auto-adopted from the checkpoint's persisted model_config, same pattern
generate.py already uses; a disagreeing CLI architecture flag is a logged
warning, not an error, since the checkpoint's values always win. Mutually
exclusive with --resume (enforced by argparse at parse time). --dataset
gains smoltalk/no_robots choices.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `generate.py` — stop decoding on the `[PAD]` signal

**Files:**
- Modify: `src/llmtrain/generate.py`
- Test: Modify `tests/test_generate.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks (this is independent of Tasks 1-4's data/training changes).
- Produces: `generate_token_ids` stops appending once a sampled token equals `pad_id`; no-op for pure-pretraining checkpoints where `[PAD]` is never sampled.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_generate.py`:

```python
class _StopsAtPadModel(torch.nn.Module):
    """Always emits pad_id as position N's argmax, real tokens elsewhere."""

    def __init__(self, vocab_size: int, pad_id: int, stop_at: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.stop_at = stop_at
        self.calls = 0
        self._unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, cache=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -10.0)
        chosen_id = self.pad_id if self.calls >= self.stop_at else 0
        logits[:, -1, chosen_id] = 10.0
        self.calls += 1
        return logits

    def parameters(self, recurse=True):
        return iter([self._unused])


def test_generate_token_ids_stops_at_pad_token_and_does_not_append_it():
    tokenizer = train_tokenizer(["hello world", "hello there"], vocab_size=32)
    pad_id = tokenizer.token_to_id("[PAD]")
    model = _StopsAtPadModel(vocab_size=tokenizer.get_vocab_size(), pad_id=pad_id, stop_at=2)
    prompt_ids = tokenizer.encode("hello").ids

    output_ids = generate_token_ids(
        model, tokenizer, "hello", GenerationConfig(max_new_tokens=5, temperature=0.0)
    )

    assert len(output_ids) == len(prompt_ids) + 2
    assert pad_id not in output_ids
```

Add `torch.nn` usage is already covered by the existing `import torch` at the top of the file — no new imports needed there.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py -v -k stops_at_pad`
Expected: FAIL — `assert 7 == 2 + 2` (i.e. `len(output_ids) == len(prompt_ids) + 5`, generation runs to `max_new_tokens` without stopping)

- [ ] **Step 3: Write the implementation**

In `src/llmtrain/generate.py`, add the `PAD_TOKEN` import:

```python
from llmtrain.data.tokenizer import PAD_TOKEN
```

In `generate_token_ids`, add the `pad_id` lookup right after `model.eval()` is entered (inside the `try:` block, before `cache = KVCache()`):

```python
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pad_id = tokenizer.token_to_id(PAD_TOKEN)

        cache = KVCache()
        generated_ids = list(prompt_ids)
```

Replace the decode loop:

```python
        with torch.no_grad():
            logits = model(input_ids, cache=cache)
            next_id = sample_next(logits[:, -1, :])
            generated_ids.append(next_id)
            for _ in range(config.max_new_tokens - 1):
                step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits = model(step_input, cache=cache)
                next_id = sample_next(logits[:, -1, :])
                generated_ids.append(next_id)
```

with:

```python
        with torch.no_grad():
            logits = model(input_ids, cache=cache)
            next_id = sample_next(logits[:, -1, :])
            if next_id != pad_id:
                generated_ids.append(next_id)
                for _ in range(config.max_new_tokens - 1):
                    step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    logits = model(step_input, cache=cache)
                    next_id = sample_next(logits[:, -1, :])
                    if next_id == pad_id:
                        break
                    generated_ids.append(next_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate.py -v`
Expected: PASS (all tests, including the new one and all pre-existing ones — pure-pretraining tests never sample `[PAD]` since it's never a supervised target there, so this is a no-op for them)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check . && uv run mypy src/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add src/llmtrain/generate.py tests/test_generate.py
git commit -m "$(cat <<'EOF'
Stop generation on the [PAD] end-of-turn signal

SFT-trained checkpoints learn to emit [PAD] right after an assistant
turn (see data/chat.py); generate_token_ids now breaks the decode loop
there instead of appending it and continuing to max_new_tokens. No-op for
pure-pretraining checkpoints, which never sample [PAD] since it's never a
supervised target in that setting.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Manual end-to-end smoke test

**Files:** none (verification only, per CLAUDE.md's convention that `train()`/`main()` orchestration has no automated test).

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: confidence that `--init-from-checkpoint` + chat data + pad-stop decoding actually work together, not just in isolated unit tests.

- [ ] **Step 1: Produce a tiny local "pretraining" checkpoint to init from**

Run (from the repo root):

```bash
rm -rf /tmp/llmtrain-sft-smoke-pretrain
uv run python -m llmtrain.training.train --dataset tiny_shakespeare --shuffle-buffer-size 50 \
  --max-steps 4 --batch-size 4 --gradient-accumulation-steps 2 \
  --d-model 32 --n-layers 2 --n-heads 2 --n-kv-heads 1 \
  --checkpoint-dir /tmp/llmtrain-sft-smoke-pretrain --wandb-mode disabled --no-compile \
  --eval-interval 4 --checkpoint-interval 4
```

Expected: completes, logs `saved checkpoint at step 4`, and `/tmp/llmtrain-sft-smoke-pretrain/step_4.pt` + `tokenizer.json` exist.

- [ ] **Step 2: Run a short SFT stage from that checkpoint**

```bash
rm -rf /tmp/llmtrain-sft-smoke-sft
uv run python -m llmtrain.training.train --dataset no_robots --shuffle-buffer-size 50 \
  --max-steps 4 --batch-size 2 --gradient-accumulation-steps 2 \
  --checkpoint-dir /tmp/llmtrain-sft-smoke-sft --wandb-mode disabled --no-compile \
  --eval-interval 4 --checkpoint-interval 4 \
  --init-from-checkpoint /tmp/llmtrain-sft-smoke-pretrain/step_4.pt
```

Expected: logs `initialized weights from checkpoint ...` (not `resumed from checkpoint`), architecture flags are silently adopted from the pretraining checkpoint (no `--d-model` etc. passed here — confirms `find_model_config_overrides` doesn't fire a spurious warning when nothing was explicitly overridden), `val_loss` is finite, and `/tmp/llmtrain-sft-smoke-sft/step_4.pt` is produced.

- [ ] **Step 3: Confirm generate.py stops before max_new_tokens at least some of the time**

```bash
for i in 1 2 3; do
  uv run python -m llmtrain.generate --checkpoint /tmp/llmtrain-sft-smoke-sft/step_4.pt \
    --prompt "Hello, how are you?" --max-new-tokens 40 --temperature 0.8
done
```

Expected: this is a smoke test, not a strict pass/fail gate — with only 4 SFT steps the model won't reliably have learned to emit `[PAD]`, so short outputs are a bonus signal, not a requirement. The run only needs to complete without error each time. If at least one of the three runs produces noticeably fewer than 40 new tokens, that's confirmation the pad-stop wiring (Task 5) is live end-to-end, not just in the unit test's fake model.

- [ ] **Step 4: Run the full automated suite one more time**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src/`
Expected: all green — this is the final gate before considering the spec done.

- [ ] **Step 5: Clean up smoke-test artifacts**

```bash
rm -rf /tmp/llmtrain-sft-smoke-pretrain /tmp/llmtrain-sft-smoke-sft
```

No commit for this task — it's verification only, nothing to stage.
