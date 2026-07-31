# Walking-Skeleton Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a working, tested, walking-skeleton training pipeline for a toy LLM, ending with `python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps N` running end-to-end on a Mac.

**Architecture:** A small `src/llmtrain` package with one module per concern (streaming data, tokenizer, model, config, checkpoint, logging) feeding a single parameterized training entry point. Every module below the GPU training loop is built test-first with tiny in-memory/CPU-only fixtures; the loop itself is validated by manually running the local smoke test.

**Tech Stack:** Python 3.12, uv, PyTorch, `tokenizers`, `datasets` (streaming), `wandb`, pytest, ruff, mypy.

## Global Constraints

- Python `>=3.12` (needed for `typing.override` used in the JSON log formatter).
- All dependency management goes through `uv` — no bare `pip install`.
- `fineweb_edu` is always loaded with `streaming=True` and `name="sample-100BT"` — never a full download.
- `DataLoader(..., pin_memory=True)` always — PyTorch itself disables it on MPS, no branching needed.
- `torch.compile` only when `device.type == "cuda"`.
- Device selection is always `torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")`.
- Stream resume uses `IterableDataset.state_dict()` / `.load_state_dict()` — no custom skip/seed tracking.
- W&B carries training metrics; the JSONL file (via `logging_config.py`'s `JSONFormatter`) carries everything else (errors, pipeline events, checkpoint events). They never overlap.
- Fail-fast TDD (failing test before implementation) for every module except the training loop orchestration itself (`train()`/CLI in Task 9), which the design explicitly validates by manual smoke test, not automated tests.
- Follow SOLID and the Karpathy anti-overengineering standard from `CLAUDE.md`: simplest thing that works, no speculative configurability.
- Full context: `docs/superpowers/specs/2026-07-31-project-scaffold-design.md`.

---

## File Structure

```
llm-training/
  pyproject.toml
  src/llmtrain/
    __init__.py
    logging_config.py         # dictConfig + JSONFormatter
    data/
      __init__.py
      tokenizer.py             # train_tokenizer, encode_batch
      streaming.py              # DATASET_REGISTRY, load_streaming_dataset
    model/
      __init__.py
      transformer.py            # MinimalTransformerLM and its sub-blocks
    training/
      __init__.py
      config.py                 # DataConfig, ModelConfig, TrainConfig
      checkpoint.py              # save_checkpoint, load_checkpoint
      train.py                    # select_device, next_token_loss, make_collate_fn, train(), main()
  tests/
    test_sanity.py
    test_logging_config.py
    test_config.py
    test_tokenizer.py
    test_streaming.py
    test_transformer.py
    test_checkpoint.py
    test_train_helpers.py
```

---

### Task 1: Project setup (uv, pyproject.toml, package skeleton)

**Files:**
- Create: `pyproject.toml`
- Create: `src/llmtrain/__init__.py`
- Create: `src/llmtrain/data/__init__.py`
- Create: `src/llmtrain/model/__init__.py`
- Create: `src/llmtrain/training/__init__.py`
- Test: `tests/test_sanity.py`

**Interfaces:**
- Produces: an installable `llmtrain` package importable from `tests/`, and `uv run pytest` / `uv run ruff check .` / `uv run mypy src/` as working commands for every later task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sanity.py
import llmtrain


def test_llmtrain_package_is_importable():
    assert llmtrain is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sanity.py -v`
Expected: FAIL (`uv` has nothing to sync yet / `ModuleNotFoundError: No module named 'llmtrain'`) — this is expected since `pyproject.toml` doesn't exist yet.

- [ ] **Step 3: Create `pyproject.toml`**

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

[dependency-groups]
dev = ["pytest>=8.1.1,<9", "ruff>=0.6", "mypy>=1.11"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llmtrain"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
```

- [ ] **Step 4: Create the package skeleton**

```bash
mkdir -p src/llmtrain/data src/llmtrain/model src/llmtrain/training
touch src/llmtrain/__init__.py src/llmtrain/data/__init__.py src/llmtrain/model/__init__.py src/llmtrain/training/__init__.py
```

- [ ] **Step 5: Sync dependencies and run the test**

Run: `uv sync && uv run pytest tests/test_sanity.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/test_sanity.py
git commit -m "chore: scaffold uv-managed llmtrain package"
```

---

### Task 2: Config dataclasses

**Files:**
- Create: `src/llmtrain/training/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `DataConfig(dataset_name: str, shuffle_buffer_size: int, max_seq_len: int, tokenizer_vocab_size: int)`, `ModelConfig(vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_seq_len: int, dropout: float)`, `TrainConfig(batch_size: int, lr: float, max_steps: int, seed: int, checkpoint_dir: str, checkpoint_interval: int, compile: bool, use_amp: bool, wandb_project: str, wandb_mode: str, log_file: str)` — all with defaults, all plain `@dataclass`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig


def test_data_config_has_sensible_defaults():
    cfg = DataConfig()
    assert cfg.dataset_name == "tiny_shakespeare"
    assert cfg.shuffle_buffer_size > 0
    assert cfg.max_seq_len > 0
    assert cfg.tokenizer_vocab_size > 0


def test_model_config_is_overridable():
    cfg = ModelConfig(d_model=64, n_layers=4)
    assert cfg.d_model == 64
    assert cfg.n_layers == 4


def test_train_config_has_sensible_defaults():
    cfg = TrainConfig()
    assert cfg.max_steps > 0
    assert cfg.batch_size > 0
    assert cfg.checkpoint_interval > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.training.config'`

- [ ] **Step 3: Write the implementation**

```python
# src/llmtrain/training/config.py
from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str = "tiny_shakespeare"
    shuffle_buffer_size: int = 1000
    max_seq_len: int = 128
    tokenizer_vocab_size: int = 1000


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_seq_len: int = 128
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 3e-4
    max_steps: int = 100
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 50
    compile: bool = True
    use_amp: bool = True
    wandb_project: str = "llm-training"
    wandb_mode: str = "online"
    log_file: str = "app.log"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/config.py tests/test_config.py
git commit -m "feat: add DataConfig/ModelConfig/TrainConfig dataclasses"
```

---

### Task 3: Logging config (JSONL + stdout)

**Files:**
- Create: `src/llmtrain/logging_config.py`
- Test: `tests/test_logging_config.py`

**Interfaces:**
- Produces: `configure_logging(log_file: str | Path = "app.log") -> None`. After calling it, `logging.getLogger(__name__)` writes INFO+ to stdout (simple format) and DEBUG+ as one JSON object per line to `log_file`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging_config.py
import json
import logging

from llmtrain.logging_config import configure_logging


def test_configure_logging_writes_parsable_jsonl_with_extra_fields(tmp_path):
    log_file = tmp_path / "test.log"
    configure_logging(log_file=log_file)

    logger = logging.getLogger("llmtrain.test_logging_config")
    logger.info("order %s received", "abc123", extra={"order_id": "abc123"})
    logging.shutdown()

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["message"] == "order abc123 received"
    assert record["order_id"] == "abc123"
    assert "timestamp" in record
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.logging_config'`

- [ ] **Step 3: Write the implementation**

```python
# src/llmtrain/logging_config.py
import datetime as dt
import json
import logging
import logging.config
from pathlib import Path
from typing import Any, override

LOG_RECORD_BUILTIN_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
}


class JSONFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None) -> None:
        super().__init__()
        self.fmt_keys = fmt_keys or {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        always_fields: dict[str, Any] = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return json.dumps(message, default=str)


def configure_logging(log_file: str | Path = "app.log") -> None:
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(levelname)-8s %(name)s: %(message)s"},
            "json": {"()": JSONFormatter},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filename": str(log_file),
                "maxBytes": 10_000_000,
                "backupCount": 3,
            },
        },
        "root": {"level": "DEBUG", "handlers": ["stdout", "file"]},
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logging_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/logging_config.py tests/test_logging_config.py
git commit -m "feat: add JSONL + stdout logging configuration"
```

---

### Task 4: Tokenizer module

**Files:**
- Create: `src/llmtrain/data/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `train_tokenizer(texts: Iterable[str], vocab_size: int) -> tokenizers.Tokenizer`, `encode_batch(tokenizer: tokenizers.Tokenizer, texts: list[str], max_seq_len: int) -> torch.Tensor` of shape `(len(texts), max_seq_len)`, dtype `torch.long`. Used by Task 8's `make_collate_fn` and Task 9's `train()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenizer.py
import torch

from llmtrain.data.tokenizer import encode_batch, train_tokenizer


def test_train_tokenizer_learns_a_vocab_from_texts():
    texts = ["hello world", "hello there", "the quick brown fox"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    assert tokenizer.get_vocab_size() > 0
    assert tokenizer.token_to_id("[PAD]") is not None


def test_encode_batch_returns_fixed_length_long_tensor():
    texts = ["hello world", "hello there", "the quick brown fox jumps"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    encoded = encode_batch(tokenizer, texts, max_seq_len=6)
    assert encoded.shape == (3, 6)
    assert encoded.dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.data.tokenizer'`

- [ ] **Step 3: Write the implementation**

```python
# src/llmtrain/data/tokenizer.py
from collections.abc import Iterable

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN]


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def encode_batch(tokenizer: Tokenizer, texts: list[str], max_seq_len: int) -> torch.Tensor:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    tokenizer.enable_truncation(max_length=max_seq_len)
    tokenizer.enable_padding(pad_token=PAD_TOKEN, pad_id=pad_id, length=max_seq_len)
    encodings = tokenizer.encode_batch(texts)
    return torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/data/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: add BPE tokenizer training and batch encoding"
```

---

### Task 5: Streaming dataset module

**Files:**
- Create: `src/llmtrain/data/streaming.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `DATASET_REGISTRY: dict[str, DatasetSpec]` with keys `"tiny_shakespeare"`, `"reformer_enwik8"`, `"fineweb_edu"`; `load_streaming_dataset(dataset_name: str, seed: int, buffer_size: int, load_fn=datasets.load_dataset) -> datasets.IterableDataset`. Used by Task 9's `train()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_streaming.py
from datasets import Dataset

from llmtrain.data.streaming import DATASET_REGISTRY, load_streaming_dataset


def _fake_load_dataset(path, name, split, streaming):
    return Dataset.from_dict(
        {"text": [f"example {i}" for i in range(20)]}
    ).to_iterable_dataset(num_shards=4)


def test_fineweb_edu_registry_entry_uses_sample_100bt_config():
    spec = DATASET_REGISTRY["fineweb_edu"]
    assert spec.path == "HuggingFaceFW/fineweb-edu"
    assert spec.name == "sample-100BT"
    assert spec.split == "train"


def test_load_streaming_dataset_shuffles_and_yields_every_example():
    dataset = load_streaming_dataset(
        "tiny_shakespeare", seed=42, buffer_size=5, load_fn=_fake_load_dataset
    )
    examples = list(dataset)
    assert len(examples) == 20
    assert all("text" in example for example in examples)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.data.streaming'`

- [ ] **Step 3: Write the implementation**

```python
# src/llmtrain/data/streaming.py
from collections.abc import Callable
from dataclasses import dataclass

from datasets import IterableDataset, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    split: str


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tiny_shakespeare": DatasetSpec(path="karpathy/tiny_shakespeare", name=None, split="train"),
    "reformer_enwik8": DatasetSpec(path="google/reformer-enwik8", name=None, split="train"),
    "fineweb_edu": DatasetSpec(
        path="HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train"
    ),
}


def load_streaming_dataset(
    dataset_name: str,
    seed: int,
    buffer_size: int,
    load_fn: Callable[..., IterableDataset] = load_dataset,
) -> IterableDataset:
    spec = DATASET_REGISTRY[dataset_name]
    dataset = load_fn(spec.path, name=spec.name, split=spec.split, streaming=True)
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/data/streaming.py tests/test_streaming.py
git commit -m "feat: add streaming dataset registry and loader"
```

---

### Task 6: Minimal transformer model

**Files:**
- Create: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `ModelConfig` from `llmtrain.training.config` (Task 2).
- Produces: `MinimalTransformerLM(config: ModelConfig)`, an `nn.Module` whose `forward(input_ids: torch.Tensor[B, T]) -> torch.Tensor[B, T, vocab_size]`. Used by Task 9's `train()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transformer.py
import torch

from llmtrain.model.transformer import MinimalTransformerLM
from llmtrain.training.config import ModelConfig


def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, max_seq_len=6, dropout=0.0)


def test_forward_produces_correct_output_shape():
    model = MinimalTransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (3, 6))
    logits = model(input_ids)
    assert logits.shape == (3, 6, 16)


def test_backward_populates_gradients_for_every_parameter():
    model = MinimalTransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (2, 6))
    logits = model(input_ids)
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.model.transformer'`

- [ ] **Step 3: Write the implementation**

```python
# src/llmtrain/model/transformer.py
import torch
from torch import nn
from torch.nn import functional as F

from llmtrain.training.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(d_model, dim=2)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MinimalTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: add minimal causal transformer LM"
```

---

### Task 7: Checkpointing (model + optimizer + dataset resume state)

**Files:**
- Create: `src/llmtrain/training/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (generic `nn.Module` / `torch.optim.Optimizer`).
- Produces: `save_checkpoint(path, model, optimizer, step: int, dataset_state: dict | None = None) -> None`, `load_checkpoint(path, model, optimizer) -> tuple[int, dict | None]`. Used by Task 9's `train()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checkpoint.py
import torch
from datasets import Dataset
from torch import nn

from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()
    original_weight = model.weight.detach().clone()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=7, dataset_state={"shard": 2})

    new_model = nn.Linear(4, 2)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    step, dataset_state = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert step == 7
    assert dataset_state == {"shard": 2}
    assert torch.equal(new_model.weight, original_weight)


def test_checkpoint_preserves_iterable_dataset_resume_position(tmp_path):
    def make_dataset():
        return Dataset.from_dict({"value": list(range(10))}).to_iterable_dataset(num_shards=2)

    original = make_dataset()
    original_iter = iter(original)
    seen = [next(original_iter)["value"] for _ in range(4)]
    state = original.state_dict()

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=4, dataset_state=state)

    new_model = nn.Linear(1, 1)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    _, restored_state = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    resumed = make_dataset()
    resumed.load_state_dict(restored_state)
    remaining = [example["value"] for example in resumed]

    assert set(seen).isdisjoint(remaining)
    assert len(seen) + len(remaining) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.training.checkpoint'`

- [ ] **Step 3: Write the implementation**

```python
# src/llmtrain/training/checkpoint.py
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset_state: dict | None = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "dataset_state": dataset_state,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict | None]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["step"], checkpoint["dataset_state"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add checkpoint save/load with dataset resume state"
```

---

### Task 8: Training helpers (device selection, loss, collate)

**Files:**
- Create: `src/llmtrain/training/train.py` (helpers only — `train()`/CLI come in Task 9)
- Test: `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `encode_batch`, `train_tokenizer` from `llmtrain.data.tokenizer` (Task 4).
- Produces: `select_device() -> torch.device`, `next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor`, `make_collate_fn(tokenizer, max_seq_len: int) -> Callable[[list[dict]], torch.Tensor]`. Used by Task 9's `train()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_helpers.py
import torch

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.training.train import make_collate_fn, next_token_loss, select_device


def test_select_device_returns_a_torch_device():
    device = select_device()
    assert isinstance(device, torch.device)


def test_next_token_loss_is_near_zero_for_perfect_predictions():
    vocab_size = 4
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, vocab_size), -100.0)
    for position, target_id in enumerate(input_ids[0, 1:]):
        logits[0, position, target_id] = 100.0
    loss = next_token_loss(logits, input_ids)
    assert loss.item() < 0.01


def test_make_collate_fn_encodes_a_batch_of_examples():
    texts = ["hello world", "hello there", "the quick brown fox"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    collate = make_collate_fn(tokenizer, max_seq_len=5)
    batch = collate([{"text": "hello world"}, {"text": "hello there"}])
    assert batch.shape == (2, 5)
    assert batch.dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.training.train'`

- [ ] **Step 3: Write the implementation**

```python
# src/llmtrain/training/train.py
from collections.abc import Callable

import torch
from torch.nn import functional as F

from llmtrain.data.tokenizer import encode_batch


def select_device() -> torch.device:
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
    )


def make_collate_fn(tokenizer, max_seq_len: int) -> Callable[[list[dict]], torch.Tensor]:
    def collate(examples: list[dict]) -> torch.Tensor:
        texts = [example["text"] for example in examples]
        return encode_batch(tokenizer, texts, max_seq_len)

    return collate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/train.py tests/test_train_helpers.py
git commit -m "feat: add device selection, next-token loss, and collate helpers"
```

---

### Task 9: Wire the training loop and CLI entry point

**Files:**
- Modify: `src/llmtrain/training/train.py` (append `train()` and `main()` to the file from Task 8)

**Interfaces:**
- Consumes: `DataConfig`/`ModelConfig`/`TrainConfig` (Task 2), `configure_logging` (Task 3), `train_tokenizer` (Task 4), `load_streaming_dataset` (Task 5), `MinimalTransformerLM` (Task 6), `save_checkpoint` (Task 7), `select_device`/`next_token_loss`/`make_collate_fn` (Task 8).
- Produces: `train(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None` and `main() -> None`, runnable as `python -m llmtrain.training.train`.

This task has **no automated test** — per the design, the training loop orchestration is the one thing validated by manually running the smoke test in Task 10, not by a unit test. Every function it calls was already tested in Tasks 2–8.

- [ ] **Step 1: Append the training loop and CLI to `train.py`**

```python
# src/llmtrain/training/train.py — append below the Task 8 helpers
import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import wandb
from torch.utils.data import DataLoader

from llmtrain.data.streaming import load_streaming_dataset
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import MinimalTransformerLM
from llmtrain.training.checkpoint import save_checkpoint
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig

logger = logging.getLogger(__name__)


def train(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    configure_logging(log_file=train_cfg.log_file)
    device = select_device()
    logger.info("training on device %s", device.type, extra={"device": device.type})

    dataset = load_streaming_dataset(
        data_cfg.dataset_name, seed=train_cfg.seed, buffer_size=data_cfg.shuffle_buffer_size
    )
    sample_texts = [example["text"] for example in dataset.take(200)]
    tokenizer = train_tokenizer(sample_texts, vocab_size=data_cfg.tokenizer_vocab_size)
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    model = MinimalTransformerLM(model_cfg).to(device)
    if device.type == "cuda" and train_cfg.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len),
    )

    wandb.init(project=train_cfg.wandb_project, mode=train_cfg.wandb_mode, config=asdict(train_cfg))

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    for batch in dataloader:
        input_ids = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=train_cfg.use_amp):
            logits = model(input_ids[:, :-1])
            loss = next_token_loss(logits, input_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()}, step=step)
        logger.debug(
            "step %d loss %.4f", step, loss.item(), extra={"step": step, "loss": loss.item()}
        )

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the toy LLM")
    parser.add_argument(
        "--dataset",
        choices=["tiny_shakespeare", "reformer_enwik8", "fineweb_edu"],
        default="tiny_shakespeare",
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    data_cfg = DataConfig(dataset_name=args.dataset)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr)
    train(data_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
```

Also add one more import this needs at the top of the file, alongside the Task 8 imports (`torch` is already imported from Task 8 — only `train_tokenizer` is new):

```python
# src/llmtrain/training/train.py — add to the existing top-of-file imports
from llmtrain.data.tokenizer import train_tokenizer
```

- [ ] **Step 2: Run the full test suite to confirm nothing broke**

Run: `uv run pytest -v`
Expected: All tests from Tasks 1–8 still PASS (this task adds no new tests).

- [ ] **Step 3: Run ruff and mypy**

Run: `uv run ruff check . && uv run mypy src/`
Expected: No errors. Fix any import-order or typing issues ruff/mypy flag before committing.

- [ ] **Step 4: Commit**

```bash
git add src/llmtrain/training/train.py
git commit -m "feat: wire train() orchestration and CLI entry point"
```

---

### Task 10: Manual local smoke test (tiny_shakespeare)

**Files:** none — this is a verification task, not a code task.

This is the plan's acceptance criterion: prove the whole pipeline actually runs end-to-end on the Mac, on real (if tiny) data, before ever touching a rented GPU.

- [ ] **Step 1: Authenticate with W&B (once)**

```bash
uv run wandb login
```

- [ ] **Step 2: Run the smoke test**

```bash
uv run python -m llmtrain.training.train --dataset tiny_shakespeare --max-steps 50 --batch-size 4
```

- [ ] **Step 3: Verify success**

Check all four:
1. The process exits with code 0 and no traceback.
2. `app.log` exists in the working directory and every line is valid JSON (`uv run python -c "import json; [json.loads(l) for l in open('app.log')]"` should raise nothing).
3. The W&B run URL printed to stdout shows a loss curve that trends downward over the 50 steps.
4. `checkpoints/` contains at least one `step_*.pt` file.

- [ ] **Step 4: Record the result**

If all four checks pass, the walking skeleton is complete — commit a short note to close out the plan:

```bash
git commit --allow-empty -m "chore: confirm tiny_shakespeare smoke test passes end-to-end"
```

If any check fails, treat it as a bug in the relevant task's module (not this task) — go back, add/adjust a unit test that would have caught it, fix, and re-run this smoke test from Step 2.
