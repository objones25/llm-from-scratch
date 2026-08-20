# Inference Serving (RunPod Serverless) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the current best DPO checkpoint (`dpo-checkpoints/step_176.pt`) as a stateless, streaming, multi-turn chat API on RunPod Serverless, so it can power a chat demo on the user's personal website.

**Architecture:** A new `src/llmtrain/serve/` module: `generation.py` holds all model/generation logic (RunPod-SDK-free, unit-testable without a real checkpoint or GPU) and `handler.py` is a thin adapter to RunPod's Python SDK. `generation.py` reuses the existing KV-cache decode loop's building blocks (`_sample`, `KVCache`, `PAD_TOKEN`) from `generate.py` rather than duplicating them, and adds multi-turn chat formatting (`format_chat_history`) to `data/chat.py`. The container is deployed to RunPod Serverless in `US-MD-1`, mounting the network volume (`08cwt7jsjv`) that already holds the checkpoint.

**Tech Stack:** Python 3.12, PyTorch, `tokenizers`, RunPod Python SDK (`runpod`, new optional dependency), Docker (CUDA base image via `uv`'s Docker pattern), RunPod Serverless.

**Spec:** `docs/superpowers/specs/2026-08-19-inference-serving-design.md`

> **Post-execution correction:** every `US-MD-1` reference below (network volume location,
> `dataCenterIds`) is **wrong** — deployment discovered `US-MD-1` has zero GPU serverless capacity
> for any GPU type despite supporting the S3 API. The real network volume and endpoint are in
> `US-IL-1`. This plan is left as originally written (a historical record of what was executed and
> why); the spec has the corrected, current values — read it, not this note's surrounding text, for
> anything DC/volume-ID-specific.

## Global Constraints

- API is stateless: the client sends the full message history on every call; no server-side session storage.
- No cross-request KV-cache persistence (no Redis, no in-memory cache keyed by session) — `model/cache.py`'s `KVCache` stays request-scoped only, exactly as it is today.
- `max_new_tokens` is hard-capped server-side at 512 regardless of what the client requests.
- Context-window overflow: truncate the oldest user/assistant turn-pairs from the front of `messages`, never mid-turn, never reject the request.
- The model's trained context ceiling is `DataConfig.max_seq_len = 2048` — this is **not** persisted in a checkpoint's saved `model_config`, so it must always be passed explicitly, never assumed recoverable from a loaded checkpoint.
- Scaling: `workersMin = 0` (scale-to-zero); `workersMax` capped low (2-3).
- The endpoint must be deployed in `US-MD-1` — the network volume holding the checkpoint (`08cwt7jsjv`) lives there, and a serverless endpoint can only mount a volume in its own data center.
- `serve/generation.py` must stay free of any RunPod SDK dependency. Only `serve/handler.py` imports `runpod`, and only inside `if __name__ == "__main__":` (so importing `handler.py` in tests never requires the `runpod` package to be installed).
- The checkpoint served is `dpo-checkpoints/step_176.pt` + `dpo-checkpoints/tokenizer.json` — the 1-epoch DPO checkpoint, README's "current best."

---

### Task 1: `format_chat_history` in `data/chat.py`

**Files:**

- Modify: `src/llmtrain/data/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**

- Produces: `format_chat_history(messages: list[dict]) -> str` — consumed by `serve/generation.py` (Tasks 4 and 6).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_chat.py`, updating the existing import line to include `format_chat_history`:

```python
from llmtrain.data.chat import (
    IGNORE_INDEX,
    encode_chat_batch,
    encode_chat_example,
    format_chat_history,
    format_prompt,
    format_turn,
)
```

Append these two tests at the end of the file:

```python
def test_format_chat_history_wraps_each_turn_and_opens_assistant_turn():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you"},
    ]
    assert format_chat_history(messages) == (
        "<|user|>\nhi\n<|assistant|>\nhello\n<|user|>\nhow are you\n<|assistant|>\n"
    )


def test_format_chat_history_single_user_turn_matches_format_prompt():
    assert format_chat_history([{"role": "user", "content": "hi"}]) == format_prompt("hi")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat.py -v`
Expected: `ImportError` (or collection error) — `format_chat_history` doesn't exist yet.

- [ ] **Step 3: Implement `format_chat_history`**

In `src/llmtrain/data/chat.py`, add this function directly after `format_prompt`:

```python
def format_chat_history(messages: list[dict]) -> str:
    # Multi-turn analogue of format_prompt: wraps every turn in its role tags, then
    # opens the trailing assistant turn the model should continue from.
    formatted = "".join(format_turn(m["role"], m["content"]) for m in messages)
    return formatted + "<|assistant|>\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/data/chat.py tests/test_chat.py
git commit -m "feat: add format_chat_history for multi-turn chat formatting"
```

---

### Task 2: `serve` package + `validate_messages`

**Files:**

- Create: `src/llmtrain/serve/__init__.py` (empty)
- Create: `src/llmtrain/serve/generation.py`
- Test: `tests/test_serve_generation.py`

**Interfaces:**

- Produces: `validate_messages(messages: list[dict] | None) -> None`, raises `ValueError` on invalid input. Consumed by `stream_chat_completion` (Task 6).

- [ ] **Step 1: Create the empty package**

```bash
mkdir -p src/llmtrain/serve
touch src/llmtrain/serve/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_serve_generation.py`:

```python
import pytest

from llmtrain.serve.generation import validate_messages


def test_validate_messages_accepts_well_formed_alternating_history():
    validate_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ]
    )  # no exception


def test_validate_messages_accepts_single_user_turn():
    validate_messages([{"role": "user", "content": "hi"}])  # no exception


def test_validate_messages_rejects_empty_list():
    with pytest.raises(ValueError):
        validate_messages([])


def test_validate_messages_rejects_none():
    with pytest.raises(ValueError):
        validate_messages(None)


def test_validate_messages_rejects_non_list():
    with pytest.raises(ValueError):
        validate_messages("not a list")


def test_validate_messages_rejects_invalid_role():
    with pytest.raises(ValueError):
        validate_messages([{"role": "system", "content": "hi"}])


def test_validate_messages_rejects_empty_content():
    with pytest.raises(ValueError):
        validate_messages([{"role": "user", "content": ""}])


def test_validate_messages_rejects_non_alternating_roles():
    with pytest.raises(ValueError):
        validate_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "hi again"},
            ]
        )


def test_validate_messages_rejects_history_ending_on_assistant_turn():
    with pytest.raises(ValueError):
        validate_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: `ModuleNotFoundError` — `llmtrain.serve.generation` doesn't exist yet.

- [ ] **Step 4: Implement `validate_messages`**

Create `src/llmtrain/serve/generation.py`:

```python
_VALID_ROLES = ("user", "assistant")


def validate_messages(messages: list[dict] | None) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    expected_role = "user"
    for i, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role not in _VALID_ROLES:
            raise ValueError(
                f"message {i} has invalid role {role!r}; must be 'user' or 'assistant'"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(f"message {i} must have non-empty string 'content'")
        if role != expected_role:
            raise ValueError(
                f"messages must alternate starting with 'user'; message {i} has role "
                f"{role!r}, expected {expected_role!r}"
            )
        expected_role = "assistant" if expected_role == "user" else "user"

    if messages[-1]["role"] != "user":
        raise ValueError("messages must end on a 'user' turn")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/llmtrain/serve/__init__.py src/llmtrain/serve/generation.py tests/test_serve_generation.py
git commit -m "feat: add serve package with chat message validation"
```

---

### Task 3: `parse_generation_config`

**Files:**

- Modify: `src/llmtrain/serve/generation.py`
- Test: `tests/test_serve_generation.py`

**Interfaces:**

- Consumes: `GenerationConfig` (`llmtrain.training.config`).
- Produces: `MAX_NEW_TOKENS_CEILING: int`, `parse_generation_config(payload: dict) -> GenerationConfig`. Consumed by `handler.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_serve_generation.py`, adding this import at the top:

```python
from llmtrain.serve.generation import (
    MAX_NEW_TOKENS_CEILING,
    parse_generation_config,
    validate_messages,
)
from llmtrain.training.config import GenerationConfig
```

```python
def test_parse_generation_config_uses_defaults_when_payload_omits_fields():
    cfg = parse_generation_config({})
    assert cfg.max_new_tokens == GenerationConfig.max_new_tokens
    assert cfg.temperature == GenerationConfig.temperature
    assert cfg.repetition_penalty == GenerationConfig.repetition_penalty
    assert cfg.top_k == GenerationConfig.top_k
    assert cfg.top_p == GenerationConfig.top_p


def test_parse_generation_config_honors_requested_values_under_ceiling():
    cfg = parse_generation_config({"max_new_tokens": 100, "temperature": 0.5})
    assert cfg.max_new_tokens == 100
    assert cfg.temperature == 0.5


def test_parse_generation_config_clamps_max_new_tokens_to_ceiling():
    cfg = parse_generation_config({"max_new_tokens": 10_000})
    assert cfg.max_new_tokens == MAX_NEW_TOKENS_CEILING
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: `ImportError` — `parse_generation_config`/`MAX_NEW_TOKENS_CEILING` don't exist yet.

- [ ] **Step 3: Implement `parse_generation_config`**

Add to `src/llmtrain/serve/generation.py` (top of file gets the new import, function goes after `validate_messages`):

```python
from llmtrain.training.config import GenerationConfig

MAX_NEW_TOKENS_CEILING = 512


def parse_generation_config(payload: dict) -> GenerationConfig:
    requested_max_new_tokens = int(payload.get("max_new_tokens", GenerationConfig.max_new_tokens))
    return GenerationConfig(
        max_new_tokens=min(requested_max_new_tokens, MAX_NEW_TOKENS_CEILING),
        temperature=float(payload.get("temperature", GenerationConfig.temperature)),
        repetition_penalty=float(
            payload.get("repetition_penalty", GenerationConfig.repetition_penalty)
        ),
        top_k=int(payload.get("top_k", GenerationConfig.top_k)),
        top_p=float(payload.get("top_p", GenerationConfig.top_p)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/serve/generation.py tests/test_serve_generation.py
git commit -m "feat: add generation-config parsing with a server-side max_new_tokens ceiling"
```

---

### Task 4: `truncate_to_context_window`

**Files:**

- Modify: `src/llmtrain/serve/generation.py`
- Test: `tests/test_serve_generation.py`

**Interfaces:**

- Consumes: `format_chat_history` (Task 1, `llmtrain.data.chat`), a `tokenizers.Tokenizer` instance.
- Produces: `truncate_to_context_window(tokenizer, messages, max_new_tokens, max_seq_len) -> list[dict]`. Consumed by `stream_chat_completion` (Task 6).

- [ ] **Step 1: Write the failing tests**

Add these imports to `tests/test_serve_generation.py`:

```python
from llmtrain.data.chat import format_chat_history
from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.serve.generation import truncate_to_context_window
```

```python
def _tiny_tokenizer():
    texts = ["hi", "hello", "how are you doing today", "I am doing well thank you"]
    return train_tokenizer(texts, vocab_size=64)


def test_truncate_returns_messages_unchanged_when_within_budget():
    tokenizer = _tiny_tokenizer()
    messages = [{"role": "user", "content": "hi"}]
    result = truncate_to_context_window(tokenizer, messages, max_new_tokens=10, max_seq_len=2048)
    assert result == messages


def test_truncate_drops_oldest_turn_pair_when_over_budget():
    tokenizer = _tiny_tokenizer()
    messages = [
        {"role": "user", "content": "how are you doing today"},
        {"role": "assistant", "content": "I am doing well thank you"},
        {"role": "user", "content": "hi"},
    ]
    prompt_len_full = len(tokenizer.encode(format_chat_history(messages)).ids)
    result = truncate_to_context_window(
        tokenizer, messages, max_new_tokens=10, max_seq_len=prompt_len_full + 5
    )
    assert result == [messages[-1]]


def test_truncate_raises_when_even_the_last_turn_does_not_fit():
    tokenizer = _tiny_tokenizer()
    messages = [{"role": "user", "content": "how are you doing today"}]
    with pytest.raises(ValueError):
        truncate_to_context_window(tokenizer, messages, max_new_tokens=10, max_seq_len=1)


def test_truncate_never_drops_a_turn_mid_pair():
    tokenizer = _tiny_tokenizer()
    messages = [
        {"role": "user", "content": "how are you doing today"},
        {"role": "assistant", "content": "I am doing well thank you"},
        {"role": "user", "content": "how are you doing today"},
        {"role": "assistant", "content": "I am doing well thank you"},
        {"role": "user", "content": "hi"},
    ]
    prompt_len_full = len(tokenizer.encode(format_chat_history(messages)).ids)
    result = truncate_to_context_window(
        tokenizer, messages, max_new_tokens=10, max_seq_len=prompt_len_full - 1
    )
    assert result[0]["role"] == "user"
    assert result[-1] == messages[-1]
    assert len(result) % 2 == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: `ImportError` — `truncate_to_context_window` doesn't exist yet.

- [ ] **Step 3: Implement `truncate_to_context_window`**

Add to `src/llmtrain/serve/generation.py` (new import at top, function after `parse_generation_config`):

```python
from tokenizers import Tokenizer

from llmtrain.data.chat import format_chat_history


def truncate_to_context_window(
    tokenizer: Tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    max_seq_len: int,
) -> list[dict]:
    messages = list(messages)
    while True:
        prompt_len = len(tokenizer.encode(format_chat_history(messages)).ids)
        if prompt_len + max_new_tokens <= max_seq_len:
            return messages
        if len(messages) <= 1:
            raise ValueError("prompt exceeds max_seq_len even after truncating all prior turns")
        messages = messages[2:]  # drop the oldest user/assistant turn-pair, never mid-turn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/serve/generation.py tests/test_serve_generation.py
git commit -m "feat: truncate oldest chat turns when a request exceeds the context window"
```

---

### Task 5: `load_model_and_tokenizer`

**Files:**

- Modify: `src/llmtrain/serve/generation.py`
- Test: `tests/test_serve_generation.py`

**Interfaces:**

- Consumes: `resolve_local_path`, `sibling_path` (`llmtrain.s3`); `load_checkpoint`, `load_model_config_from_checkpoint` (`llmtrain.training.checkpoint`); `TransformerLM` (`llmtrain.model.transformer`).
- Produces: `load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str | None = None) -> tuple[TransformerLM, Tokenizer]`. Consumed by `handler.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

Add these imports to `tests/test_serve_generation.py`:

```python
import torch

from llmtrain.model.transformer import TransformerLM
from llmtrain.serve.generation import load_model_and_tokenizer
from llmtrain.training.checkpoint import save_checkpoint
from llmtrain.training.config import ModelConfig
```

```python
def test_load_model_and_tokenizer_round_trips_a_saved_checkpoint(tmp_path):
    tokenizer = train_tokenizer(["hello world", "hello there"], vocab_size=32)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=8, n_layers=2, n_heads=4, n_kv_heads=2
    )
    model = TransformerLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "step_1.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    save_checkpoint(checkpoint_path, model, optimizer, step=1)
    tokenizer.save(str(tokenizer_path))

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(str(checkpoint_path))

    assert isinstance(loaded_model, TransformerLM)
    assert loaded_model.training is False
    assert loaded_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()


def test_load_model_and_tokenizer_honors_explicit_tokenizer_path(tmp_path):
    tokenizer = train_tokenizer(["hello world"], vocab_size=32)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=8, n_layers=2, n_heads=4, n_kv_heads=2
    )
    model = TransformerLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "step_1.pt"
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    tokenizer_path = other_dir / "tok.json"
    save_checkpoint(checkpoint_path, model, optimizer, step=1)
    tokenizer.save(str(tokenizer_path))

    _loaded_model, loaded_tokenizer = load_model_and_tokenizer(
        str(checkpoint_path), str(tokenizer_path)
    )
    assert loaded_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: `ImportError` — `load_model_and_tokenizer` doesn't exist yet.

- [ ] **Step 3: Implement `load_model_and_tokenizer`**

Add to `src/llmtrain/serve/generation.py` (new imports at top, function after `truncate_to_context_window`):

```python
from llmtrain.model.transformer import TransformerLM
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import load_checkpoint, load_model_config_from_checkpoint


def load_model_and_tokenizer(
    checkpoint_path: str, tokenizer_path: str | None = None
) -> tuple[TransformerLM, Tokenizer]:
    tokenizer_uri = tokenizer_path or sibling_path(checkpoint_path, "tokenizer.json")
    resolved_checkpoint = resolve_local_path(checkpoint_path)
    resolved_tokenizer = resolve_local_path(tokenizer_uri)

    tokenizer = Tokenizer.from_file(str(resolved_tokenizer))
    model_cfg = load_model_config_from_checkpoint(resolved_checkpoint, tokenizer.get_vocab_size())
    model = TransformerLM(model_cfg)
    load_checkpoint(resolved_checkpoint, model)
    model.eval()
    return model, tokenizer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/serve/generation.py tests/test_serve_generation.py
git commit -m "feat: load model and tokenizer from a checkpoint path for serving"
```

---

### Task 6: `stream_chat_completion`

**Files:**

- Modify: `src/llmtrain/serve/generation.py`
- Test: `tests/test_serve_generation.py`

**Interfaces:**

- Consumes: `validate_messages`, `truncate_to_context_window` (this file); `format_chat_history` (`llmtrain.data.chat`); `_sample` (`llmtrain.generate`); `KVCache` (`llmtrain.model.cache`); `PAD_TOKEN` (`llmtrain.data.tokenizer`); `GenerationConfig` (`llmtrain.training.config`).
- Produces: `stream_chat_completion(model, tokenizer, messages, generation_cfg, max_seq_len) -> Iterator[str]`. Consumed by `handler.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

Add these imports to `tests/test_serve_generation.py`:

```python
from collections.abc import Iterator

from llmtrain.data.tokenizer import PAD_TOKEN
from llmtrain.generate import generate_token_ids
from llmtrain.serve.generation import stream_chat_completion
```

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


def test_stream_chat_completion_yields_text_and_stops_at_pad_token():
    tokenizer = train_tokenizer(["hello world", "hello there"], vocab_size=32)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    model = _StopsAtPadModel(vocab_size=tokenizer.get_vocab_size(), pad_id=pad_id, stop_at=2)
    messages = [{"role": "user", "content": "hello"}]

    chunks = list(
        stream_chat_completion(
            model,
            tokenizer,
            messages,
            GenerationConfig(max_new_tokens=5, temperature=0.0),
            max_seq_len=2048,
        )
    )

    assert len(chunks) == 2
    assert all(isinstance(c, str) and c for c in chunks)


def test_stream_chat_completion_matches_generate_token_ids_output():
    torch.manual_seed(0)
    tokenizer = train_tokenizer(
        ["hello world", "hello there", "world hello there"], vocab_size=32
    )
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=8, n_layers=2, n_heads=4, n_kv_heads=2
    )
    model = TransformerLM(config)
    generation_cfg = GenerationConfig(max_new_tokens=5, temperature=0.0)
    messages = [{"role": "user", "content": "hello"}]

    streamed_text = "".join(
        stream_chat_completion(model, tokenizer, messages, generation_cfg, max_seq_len=2048)
    )

    prompt = format_chat_history(messages)
    expected_ids = generate_token_ids(model, tokenizer, prompt, generation_cfg)
    prompt_len = len(tokenizer.encode(prompt).ids)
    expected_text = tokenizer.decode(expected_ids[prompt_len:])

    assert streamed_text == expected_text


def test_stream_chat_completion_validates_messages_before_any_model_call():
    tokenizer = train_tokenizer(["hello world"], vocab_size=32)
    model = _StopsAtPadModel(vocab_size=tokenizer.get_vocab_size(), pad_id=0, stop_at=0)
    with pytest.raises(ValueError):
        list(
            stream_chat_completion(
                model, tokenizer, [], GenerationConfig(max_new_tokens=5), max_seq_len=2048
            )
        )
    assert model.calls == 0


def test_stream_chat_completion_yields_nothing_when_max_new_tokens_is_zero():
    tokenizer = train_tokenizer(["hello world"], vocab_size=32)
    model = _StopsAtPadModel(vocab_size=tokenizer.get_vocab_size(), pad_id=0, stop_at=0)
    messages = [{"role": "user", "content": "hello"}]
    chunks = list(
        stream_chat_completion(
            model, tokenizer, messages, GenerationConfig(max_new_tokens=0), max_seq_len=2048
        )
    )
    assert chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: `ImportError` — `stream_chat_completion` doesn't exist yet.

- [ ] **Step 3: Implement `stream_chat_completion`**

Add to `src/llmtrain/serve/generation.py` (new imports at top, function after `load_model_and_tokenizer`):

```python
from collections.abc import Iterator

import torch

from llmtrain.data.tokenizer import PAD_TOKEN
from llmtrain.generate import _sample
from llmtrain.model.cache import KVCache


def stream_chat_completion(
    model: TransformerLM,
    tokenizer: Tokenizer,
    messages: list[dict],
    generation_cfg: GenerationConfig,
    max_seq_len: int,
) -> Iterator[str]:
    validate_messages(messages)
    messages = truncate_to_context_window(
        tokenizer, messages, generation_cfg.max_new_tokens, max_seq_len
    )
    prompt_ids = tokenizer.encode(format_chat_history(messages)).ids
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    if generation_cfg.max_new_tokens <= 0:
        return

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        cache = KVCache(max_seq_len=len(prompt_ids) + generation_cfg.max_new_tokens)
        generated_ids = list(prompt_ids)
        new_ids: list[int] = []
        prev_text = ""

        def sample_next(logits: torch.Tensor) -> int:
            return _sample(
                logits,
                generated_ids,
                generation_cfg.temperature,
                generation_cfg.repetition_penalty,
                generation_cfg.top_k,
                generation_cfg.top_p,
            )

        def emit(token_id: int) -> str:
            nonlocal prev_text
            new_ids.append(token_id)
            generated_ids.append(token_id)
            full_text = tokenizer.decode(new_ids)
            delta = full_text[len(prev_text) :]
            prev_text = full_text
            return delta

        with torch.no_grad():
            logits = model(input_ids, cache=cache)
            next_id = sample_next(logits[:, -1, :])
            if next_id != pad_id:
                delta = emit(next_id)
                if delta:
                    yield delta
                for _ in range(generation_cfg.max_new_tokens - 1):
                    step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    logits = model(step_input, cache=cache)
                    next_id = sample_next(logits[:, -1, :])
                    if next_id == pad_id:
                        break
                    delta = emit(next_id)
                    if delta:
                        yield delta
    finally:
        model.train(was_training)
```

**Why decode-diff instead of decoding each new token in isolation:** this project's tokenizer is byte-level BPE, so a single multi-byte UTF-8 character can be split across two token ids — decoding a lone new token id in isolation can produce mangled text at that boundary. Decoding the full `new_ids` list so far and yielding only the substring past what was already emitted (the same trick `transformers.TextStreamer` uses) avoids that, and `test_stream_chat_completion_matches_generate_token_ids_output` above confirms it round-trips identically to batch decoding.

**Note on generators:** `stream_chat_completion` contains `yield`, so calling it does not run any code (not even `validate_messages`) until the result is iterated — this is standard Python generator behavior. `handler.py`'s `for token_text in generation.stream_chat_completion(...)` (Task 7) iterates it immediately, so validation still happens before any model call in practice.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/serve/generation.py tests/test_serve_generation.py
git commit -m "feat: stream chat completions from a loaded checkpoint"
```

---

### Task 7: RunPod handler adapter (`serve/handler.py`)

**Files:**

- Modify: `pyproject.toml`
- Create: `src/llmtrain/serve/handler.py`
- Test: `tests/test_serve_handler.py`

**Interfaces:**

- Consumes: `generation.parse_generation_config`, `generation.stream_chat_completion`, `generation.load_model_and_tokenizer` (Tasks 3, 5, 6); `DataConfig.max_seq_len` (`llmtrain.training.config`).
- Produces: `handler(job: dict) -> Iterator[dict]`, `_get_model_and_tokenizer() -> tuple[TransformerLM, Tokenizer]` (module-level, monkeypatchable).

- [ ] **Step 1: Add the `serve` optional dependency**

In `pyproject.toml`, under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
cuda = ["liger-kernel>=0.8"]
s3 = ["boto3>=1.34"]
serve = ["runpod>=1.7"]
```

Run: `uv sync --extra serve`
Expected: resolves and installs `runpod` cleanly, updates `uv.lock`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_serve_handler.py`:

```python
import pytest

from llmtrain.serve import handler


@pytest.fixture(autouse=True)
def _reset_model_cache():
    handler._model = None
    handler._tokenizer = None
    yield
    handler._model = None
    handler._tokenizer = None


def test_get_model_and_tokenizer_loads_once_and_caches(monkeypatch):
    calls = []

    def fake_load(checkpoint_path, tokenizer_path):
        calls.append((checkpoint_path, tokenizer_path))
        return "fake-model", "fake-tokenizer"

    monkeypatch.setattr(handler.generation, "load_model_and_tokenizer", fake_load)

    first = handler._get_model_and_tokenizer()
    second = handler._get_model_and_tokenizer()

    assert first == ("fake-model", "fake-tokenizer")
    assert second == first
    assert len(calls) == 1


def test_handler_yields_streamed_tokens_then_done(monkeypatch):
    monkeypatch.setattr(handler, "_get_model_and_tokenizer", lambda: ("fake-model", "fake-tok"))
    monkeypatch.setattr(
        handler.generation, "stream_chat_completion", lambda *a, **k: iter(["He", "llo"])
    )

    job = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    chunks = list(handler.handler(job))

    assert chunks == [
        {"token": "He", "done": False},
        {"token": "llo", "done": False},
        {"done": True},
    ]


def test_handler_yields_structured_error_on_invalid_input(monkeypatch):
    monkeypatch.setattr(handler, "_get_model_and_tokenizer", lambda: ("fake-model", "fake-tok"))

    def raise_value_error(*args, **kwargs):
        raise ValueError("messages must be a non-empty list")

    monkeypatch.setattr(handler.generation, "stream_chat_completion", raise_value_error)

    job = {"input": {"messages": []}}
    chunks = list(handler.handler(job))

    assert chunks == [{"error": "messages must be a non-empty list", "done": True}]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve_handler.py -v`
Expected: `ModuleNotFoundError` — `llmtrain.serve.handler` doesn't exist yet.

- [ ] **Step 4: Implement `handler.py`**

Create `src/llmtrain/serve/handler.py`:

```python
import logging
import os
from collections.abc import Iterator

from llmtrain.logging_config import configure_logging
from llmtrain.serve import generation
from llmtrain.training.config import DataConfig

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "/runpod-volume/dpo-checkpoints/step_176.pt"
)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = generation.load_model_and_tokenizer(CHECKPOINT_PATH, TOKENIZER_PATH)
    return _model, _tokenizer


def handler(job: dict) -> Iterator[dict]:
    model, tokenizer = _get_model_and_tokenizer()
    payload = job.get("input", {})
    messages = payload.get("messages")
    generation_cfg = generation.parse_generation_config(payload)

    try:
        for token_text in generation.stream_chat_completion(
            model, tokenizer, messages, generation_cfg, DataConfig.max_seq_len
        ):
            yield {"token": token_text, "done": False}
    except ValueError as exc:
        logger.warning(f"rejected invalid request: {exc}")
        yield {"error": str(exc), "done": True}
        return

    yield {"done": True}


if __name__ == "__main__":
    import runpod

    configure_logging()
    logger.info(f"loading model and tokenizer from {CHECKPOINT_PATH}")
    _get_model_and_tokenizer()  # load once at process start, before accepting jobs
    logger.info("model loaded, starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
```

`import runpod` is deferred inside the `__main__` guard, matching `s3.py`'s lazy-import-of-`boto3` pattern — importing `handler.py` in tests never requires the `runpod` package, and `generation.py` (Tasks 2-6) never imports it at all. `configure_logging()` is only called here (not at module import time) so tests importing `handler.py` don't redirect logging as a side effect; an uncaught exception during `_get_model_and_tokenizer()` or inside `handler()` (e.g. a CUDA OOM) still gets Python's normal traceback plus whatever was already logged via `logger`, and is otherwise left to propagate — RunPod's SDK catches it and marks the job/worker failed, which is the intended behavior for anything that isn't the explicitly-handled `ValueError` validation case.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve_handler.py -v`
Expected: PASS

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest`
Expected: PASS (all tests, including every prior task's)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/llmtrain/serve/handler.py tests/test_serve_handler.py
git commit -m "feat: add RunPod serverless handler adapter"
```

---

### Task 8: Dockerfile

**Files:**

- Create: `Dockerfile` (repo root)

**Interfaces:**

- Consumes: `src/llmtrain/serve/handler.py` (Task 7) as the container's entry point.
- Produces: a buildable image tagged `llm-training-serve:test` locally, later pushed to `ghcr.io/objones25/llm-training-serve` in Task 9.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv python install 3.12 && \
    uv sync --frozen --extra cuda --extra serve --extra s3 --no-dev

ENV CHECKPOINT_PATH=/runpod-volume/dpo-checkpoints/step_176.pt
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "-m", "llmtrain.serve.handler"]
```

Ubuntu 22.04's default apt repos don't ship Python 3.12, so this uses `uv`'s own documented Docker pattern instead of an apt-installed interpreter: copy the `uv`/`uvx` binaries from the official `uv` image, then let `uv python install 3.12` fetch a managed interpreter regardless of the base OS. `--frozen` makes the build fail loudly if `uv.lock` and `pyproject.toml` have drifted, rather than silently re-resolving.

- [ ] **Step 2: Verify the image builds**

Run: `docker build -t llm-training-serve:test .`
Expected: build succeeds. This only needs Docker installed locally — it does not require a GPU or CUDA driver, since building never runs the CUDA runtime, only installs it.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for the RunPod serverless container"
```

---

### Task 9: Deploy the RunPod Serverless endpoint

**Files:** none (infrastructure only, via RunPod's MCP tools / `docker` CLI)

**Interfaces:**

- Consumes: the image built in Task 8; the network volume `08cwt7jsjv` (`US-MD-1`) created earlier holding `dpo-checkpoints/step_176.pt` + `tokenizer.json`.
- Produces: a live RunPod Serverless endpoint URL — the integration point the website's proxy (separate repo, out of scope per the spec) will call.

- [ ] **Step 1: Push the image to GitHub Container Registry**

```bash
docker login ghcr.io -u objones25   # password: a GitHub PAT with write:packages scope
docker tag llm-training-serve:test ghcr.io/objones25/llm-training-serve:latest
docker push ghcr.io/objones25/llm-training-serve:latest
```

Then, in the GitHub UI (`github.com/objones25` → Packages → `llm-training-serve` → Package settings), set the package visibility to **public**. This lets RunPod pull the image without needing a `containerRegistryAuthId` credential — simplest option for a personal project with nothing sensitive baked into the image itself (the checkpoint is mounted from the network volume at runtime, never baked into the image).

- [ ] **Step 2: Re-check GPU availability**

GPU stock and pricing are live and may have shifted since this plan was written. Re-run before creating the endpoint:

Call `list-gpu-types` with `minMemoryGb: 16`, `product: "SERVERLESS"`, `includeUnavailable: false`.

At plan-writing time, the cheapest `HIGH`-availability option meeting the model's needs (weights are ~0.95GB at fp16, comfortably fits any of these) was `NVIDIA RTX A5000` (pool `AMPERE_24`, $0.69/hr serverless, 24GB). If a cheaper `HIGH`-availability option now exists, use its `pool` value instead.

- [ ] **Step 3: Create the endpoint**

Call `create-endpoint` with:

```json
{
  "name": "llm-inference-chat",
  "imageName": "ghcr.io/objones25/llm-training-serve:latest",
  "gpuPoolIds": ["AMPERE_24"],
  "networkVolumeIds": ["08cwt7jsjv"],
  "dataCenterIds": ["US-MD-1"],
  "workersMin": 0,
  "workersMax": 3,
  "executionTimeoutMs": 120000,
  "idleTimeout": 300,
  "containerDiskInGb": 15
}
```

`dataCenterIds` must be `US-MD-1` — that's where the network volume lives, and a serverless endpoint can only mount a volume in its own data center. `idleTimeout: 300` (5 minutes) keeps a worker warm across a typical multi-turn conversation's back-and-forth without repeated cold starts mid-chat, while still scaling to zero well before real idle cost accrues.

Record the endpoint id and the `requestUrls` from the response — the website proxy needs these.

- [ ] **Step 4: Manual smoke test against the real deployed endpoint**

Call `runsync-endpoint` (or `run-endpoint` + poll `get-job-status`) with:

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "What's the capital of France?" }
    ],
    "max_new_tokens": 50,
    "temperature": 0.0
  }
}
```

Expected: the job completes, the aggregated stream output is a sequence of `{"token": ..., "done": false}` chunks ending in `{"done": true}`, and the concatenated tokens read as a coherent, on-topic answer (matching the quality already confirmed in `docs/dpo-run-results.md` for this exact checkpoint). Note the cold-start time from job submission to first token — this is the real number to report back, not the "10-60s" estimate from the design spec.

This step needs a real GPU and the real checkpoint, so — consistent with this project's existing convention — it's a manual smoke test, not an automated one.

- [ ] **Step 5: No commit** — this task is pure infrastructure provisioning, nothing in the repo changes.

---

### Task 10: Document the `serve` module in `CLAUDE.md`

**Files:**

- Modify: `CLAUDE.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Add `serve/` to the Architecture file tree**

In `CLAUDE.md`'s `src/llmtrain/` architecture listing, add after the `s3.py` entry:

```
serve/generation.py    # RunPod-SDK-free inference core: validate_messages, truncate_to_context_window,
                       # parse_generation_config, load_model_and_tokenizer, stream_chat_completion --
                       # reuses generate.py's _sample/KVCache/PAD_TOKEN rather than duplicating them
serve/handler.py        # Thin RunPod Python SDK adapter (runpod.serverless.start) -- parses job
                       # input, calls generation.stream_chat_completion, yields RunPod's streaming
                       # chunk shape. Imports runpod lazily, only inside `if __name__ == "__main__"`,
                       # so generation.py and this file's own module scope stay runpod-free for tests.
```

- [ ] **Step 2: Add a short prose section**

After the DPO pipeline section, add:

```markdown
## Inference serving

`src/llmtrain/serve/` serves a checkpoint (currently `dpo-checkpoints/step_176.pt`) as a stateless,
streaming, multi-turn chat API on RunPod Serverless — see
`docs/superpowers/specs/2026-08-19-inference-serving-design.md` for the full design (why
stateless, why no cross-request KV-cache, deployment config) and
`docs/superpowers/plans/2026-08-19-inference-serving.md` for the implementation plan. The API is
stateless by design: the client resends the full message history every call, and
`model/cache.py`'s `KVCache` stays request-scoped, never persisted across requests.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the serve module in CLAUDE.md"
```
