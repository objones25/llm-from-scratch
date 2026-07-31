# Architecture Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `model/transformer.py`'s GPT-2-style components (learned position embeddings, LayerNorm, GELU MLP, plain MHA) with RMSNorm, RoPE, SwiGLU, weight tying, and GQA, and add KV-cache-backed generation via a new `generate.py`.

**Architecture:** All changes are additive/substitutive within `MinimalTransformerLM`'s existing three-class structure (`CausalSelfAttention`, `MLP`, `Block`, `MinimalTransformerLM`), plus one new `model/cache.py` module and one new top-level `generate.py` entry point (mirroring the existing flat-module layout, same level as `logging_config.py`). No changes to `data/`, `training/checkpoint.py`, or the training loop's control flow — only `training/config.py` gains two new `ModelConfig` fields, and `training/train.py` gains one line to persist the tokenizer alongside checkpoints (needed for `generate.py` to be usable against a real trained model).

**Tech Stack:** PyTorch >=2.6 (`torch.nn.RMSNorm`, `F.scaled_dot_product_attention(..., enable_gqa=True)`), `tokenizers` (HF `Tokenizer.save`/`Tokenizer.from_file`), pytest.

## Global Constraints

- Breaking change: not checkpoint-compatible with the current architecture. No migration path — only smoke-test checkpoints exist so far.
- `torch.nn.RMSNorm` is used directly (native since torch 2.4) — do not hand-roll RMSNorm.
- RoPE cos/sin are computed on the fly per forward call, sized to the actual `seq_len` passed in — never precomputed/cached to a fixed `max_seq_len`. This is what removes the hard context-length ceiling a learned embedding table bakes in.
- `F.scaled_dot_product_attention(..., enable_gqa=True)` is used directly for GQA — empirically verified to work on both CPU and MPS with the installed torch 2.13.0 (see spec's verification notes). No manual `repeat_interleave` fallback is implemented; the GQA compatibility test is the regression guard if a future torch version breaks this.
- `is_causal=True` is correct for both the multi-token prefill step and the single-token cached-decode step (query length 1, key length N) — verified against current PyTorch docs, no manual attention mask is built anywhere in this plan.
- Every new/changed piece of model code gets a real, fast, CPU-only unit test with tiny fake data — no GPU, no network. `train()`/`main()` orchestration remains untested by design (existing project convention); only the one-line tokenizer-persistence addition to `train.py` is exempt from a dedicated test for that reason.

---

## Task 1: `ModelConfig` gains `rope_theta` and `n_kv_heads`

**Files:**
- Modify: `src/llmtrain/training/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ModelConfig.rope_theta: float` (default `10000.0`), `ModelConfig.n_kv_heads: int` (default `1`) — consumed by Tasks 5 and 6.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_model_config_has_rope_and_gqa_defaults():
    cfg = ModelConfig()
    assert cfg.rope_theta == 10000.0
    assert cfg.n_kv_heads == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_model_config_has_rope_and_gqa_defaults -v`
Expected: FAIL with `TypeError: ModelConfig.__init__() got an unexpected keyword argument` or `AttributeError` (the fields don't exist yet).

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/training/config.py`, add two fields to `ModelConfig`:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 1000
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 1
    max_seq_len: int = 128
    dropout: float = 0.0
    rope_theta: float = 10000.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all tests in the file, including the two pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/config.py tests/test_config.py
git commit -m "feat: add rope_theta and n_kv_heads to ModelConfig"
```

---

## Task 2: Swap `nn.LayerNorm` for `nn.RMSNorm`

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no interface change — `Block.ln1`, `Block.ln2`, `MinimalTransformerLM.ln_f` remain the same attribute names, just a different `nn.Module` subclass.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transformer.py`:

```python
from torch import nn


def test_uses_rmsnorm_not_layernorm():
    model = MinimalTransformerLM(_tiny_config())
    assert isinstance(model.blocks[0].ln1, nn.RMSNorm)
    assert isinstance(model.blocks[0].ln2, nn.RMSNorm)
    assert isinstance(model.ln_f, nn.RMSNorm)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py::test_uses_rmsnorm_not_layernorm -v`
Expected: FAIL with `AssertionError` (currently `nn.LayerNorm` instances).

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, in `Block.__init__`, replace:

```python
        self.ln1 = nn.LayerNorm(config.d_model)
```
with
```python
        self.ln1 = nn.RMSNorm(config.d_model)
```

and
```python
        self.ln2 = nn.LayerNorm(config.d_model)
```
with
```python
        self.ln2 = nn.RMSNorm(config.d_model)
```

In `MinimalTransformerLM.__init__`, replace:
```python
        self.ln_f = nn.LayerNorm(config.d_model)
```
with
```python
        self.ln_f = nn.RMSNorm(config.d_model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS (all tests, including the two pre-existing shape/gradient tests)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: replace LayerNorm with native RMSNorm"
```

---

## Task 3: Weight tying

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `model.head.weight is model.token_emb.weight` holds for every `MinimalTransformerLM` instance.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transformer.py`:

```python
def test_head_weight_is_tied_to_token_embedding():
    model = MinimalTransformerLM(_tiny_config())
    assert model.head.weight is model.token_emb.weight


def test_tied_weight_receives_gradient():
    model = MinimalTransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (2, 6))
    logits = model(input_ids)
    logits.sum().backward()
    assert model.token_emb.weight.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py::test_head_weight_is_tied_to_token_embedding -v`
Expected: FAIL with `AssertionError` (currently two separate weight tensors).

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, in `MinimalTransformerLM.__init__`, after:

```python
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
```

add:

```python
        self.head.weight = self.token_emb.weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: tie token embedding and output head weights"
```

---

## Task 4: SwiGLU MLP

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `MLP` gains attributes `w_gate`, `w_up`, `w_down` (all `nn.Linear`), replacing the old `net` attribute. `MLP.forward(x) -> Tensor` keeps the same signature and output shape.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transformer.py`:

```python
from llmtrain.model.transformer import MLP


def test_mlp_output_shape_matches_input():
    config = _tiny_config()
    mlp = MLP(config)
    x = torch.randn(3, 6, config.d_model)
    out = mlp(x)
    assert out.shape == x.shape


def test_mlp_uses_swiglu_hidden_dim_formula():
    config = _tiny_config()
    mlp = MLP(config)
    expected_d_ff = int(2 / 3 * 4 * config.d_model)
    assert mlp.w_gate.out_features == expected_d_ff
    assert mlp.w_up.out_features == expected_d_ff
    assert mlp.w_down.in_features == expected_d_ff
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py::test_mlp_uses_swiglu_hidden_dim_formula -v`
Expected: FAIL with `AttributeError: 'MLP' object has no attribute 'w_gate'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, replace the `MLP` class body:

```python
class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_ff = int(2 / 3 * 4 * config.d_model)
        self.w_gate = nn.Linear(config.d_model, d_ff)
        self.w_up = nn.Linear(config.d_model, d_ff)
        self.w_down = nn.Linear(d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: replace GELU MLP with SwiGLU"
```

---

## Task 5: RoPE

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `ModelConfig.rope_theta` (Task 1).
- Produces: module-level functions `_rotary_cos_sin(seq_len: int, head_dim: int, theta: float, position_offset: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]` and `apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor` in `llmtrain.model.transformer` — consumed by Task 6 and Task 8. `CausalSelfAttention.forward` gains a `position_offset: int = 0` keyword parameter — consumed by Task 8. `Block.forward` gains the same parameter, threading it through. `MinimalTransformerLM` no longer has a `pos_emb` attribute.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transformer.py`:

```python
from llmtrain.model.transformer import _rotary_cos_sin, apply_rotary


def test_rope_is_identity_at_position_zero():
    head_dim = 4
    cos, sin = _rotary_cos_sin(
        seq_len=1,
        head_dim=head_dim,
        theta=10000.0,
        position_offset=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x = torch.randn(1, 1, 1, head_dim)
    rotated = apply_rotary(x, cos, sin)
    assert torch.allclose(rotated, x, atol=1e-6)


def test_forward_handles_seq_len_larger_than_max_seq_len_used_at_construction():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_seq_len=6, dropout=0.0
    )
    model = MinimalTransformerLM(config)
    input_ids = torch.randint(0, 16, (2, 50))
    logits = model(input_ids)
    assert logits.shape == (2, 50, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py::test_rope_is_identity_at_position_zero tests/test_transformer.py::test_forward_handles_seq_len_larger_than_max_seq_len_used_at_construction -v`
Expected: first FAILs with `ImportError` (`_rotary_cos_sin` doesn't exist yet); second currently PASSES-then-will-fail once `pos_emb` is removed — for now, if run against current code it raises `IndexError: index out of range in self` from the `pos_emb` embedding lookup once seq_len=50 > max_seq_len=6. Confirm both fail before proceeding (adjust the import at the top of the test file first so the file can even be collected).

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, add module-level helpers (after the imports, before `CausalSelfAttention`):

```python
def _rotary_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float,
    position_offset: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(
        position_offset, position_offset + seq_len, device=device, dtype=torch.float32
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin
```

Update `CausalSelfAttention`:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position embeddings")
        self.rope_theta = config.rope_theta
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(d_model, dim=2)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rotary_cos_sin(
            seq_len, self.head_dim, self.rope_theta, position_offset, x.device, x.dtype
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)
```

Update `Block`:

```python
class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), position_offset=position_offset)
        x = x + self.mlp(self.ln2(x))
        return x
```

Update `MinimalTransformerLM`: remove `self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)` from `__init__`, and in `forward`, remove the `positions`/`pos_emb` lookup:

```python
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: replace learned position embeddings with RoPE"
```

---

## Task 6: GQA

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `ModelConfig.n_kv_heads` (Task 1), `_rotary_cos_sin`/`apply_rotary` (Task 5).
- Produces: `CausalSelfAttention` gains `q_proj`, `kv_proj` attributes (replacing `qkv_proj`) and `n_kv_heads` attribute — consumed by Task 8's cache wiring, which reads K/V head count implicitly via tensor shape.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transformer.py`:

```python
from llmtrain.model.transformer import CausalSelfAttention
import pytest


def test_gqa_projections_are_sized_for_fewer_kv_heads():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=6, dropout=0.0
    )
    attn = CausalSelfAttention(config)
    assert attn.q_proj.out_features == config.n_heads * attn.head_dim
    assert attn.kv_proj.out_features == 2 * config.n_kv_heads * attn.head_dim


def test_forward_works_with_grouped_query_attention():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=6, dropout=0.0
    )
    model = MinimalTransformerLM(config)
    input_ids = torch.randint(0, 16, (2, 6))
    logits = model(input_ids)
    assert logits.shape == (2, 6, 16)
    logits.sum().backward()


def test_n_heads_must_be_divisible_by_n_kv_heads():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=4, n_kv_heads=3, max_seq_len=6, dropout=0.0
    )
    with pytest.raises(ValueError):
        CausalSelfAttention(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py::test_gqa_projections_are_sized_for_fewer_kv_heads -v`
Expected: FAIL with `AttributeError: 'CausalSelfAttention' object has no attribute 'q_proj'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, replace `CausalSelfAttention`:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position embeddings")
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim)
        self.kv_proj = nn.Linear(config.d_model, 2 * config.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x)
        k, v = kv.split(self.n_kv_heads * self.head_dim, dim=2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rotary_cos_sin(
            seq_len, self.head_dim, self.rope_theta, position_offset, x.device, x.dtype
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS (all tests, including the pre-existing ones — confirms `enable_gqa=True` works with the default `n_kv_heads=1` on this machine's CPU, which is the GQA compatibility regression guard described in Global Constraints)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: add grouped-query attention (GQA)"
```

---

## Task 7: `KVCache` class

**Files:**
- Create: `src/llmtrain/model/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `class KVCache` with `.update(layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]` (k/v shaped `(batch, n_kv_heads, seq_len, head_dim)`, concatenated along the sequence dim) and `.seq_len -> int` property — consumed by Task 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache.py`:

```python
import torch

from llmtrain.model.cache import KVCache


def test_update_returns_new_kv_when_cache_empty():
    cache = KVCache()
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    k_out, v_out = cache.update(layer_idx=0, k=k, v=v)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)
    assert cache.seq_len == 3


def test_update_concatenates_across_calls():
    cache = KVCache()
    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    cache.update(layer_idx=0, k=k1, v=v1)
    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    k_out, v_out = cache.update(layer_idx=0, k=k2, v=v2)
    assert k_out.shape == (1, 2, 4, 4)
    assert torch.equal(k_out[:, :, :3, :], k1)
    assert torch.equal(k_out[:, :, 3:, :], k2)
    assert cache.seq_len == 4


def test_layers_are_cached_independently():
    cache = KVCache()
    cache.update(layer_idx=0, k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4))
    cache.update(layer_idx=1, k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4))
    k0_out, _ = cache.update(layer_idx=0, k=torch.randn(1, 2, 1, 4), v=torch.randn(1, 2, 1, 4))
    assert k0_out.shape == (1, 2, 3, 4)


def test_seq_len_is_zero_for_empty_cache():
    cache = KVCache()
    assert cache.seq_len == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.model.cache'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/llmtrain/model/cache.py`:

```python
import torch


class KVCache:
    def __init__(self) -> None:
        self._entries: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def seq_len(self) -> int:
        if not self._entries:
            return 0
        first_k, _ = next(iter(self._entries.values()))
        return first_k.shape[2]

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self._entries:
            prev_k, prev_v = self._entries[layer_idx]
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        self._entries[layer_idx] = (k, v)
        return k, v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cache.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/cache.py tests/test_cache.py
git commit -m "feat: add KVCache for autoregressive decoding"
```

---

## Task 8: Wire `KVCache` through `CausalSelfAttention` and `MinimalTransformerLM`

**Files:**
- Modify: `src/llmtrain/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `KVCache` (Task 7).
- Produces: `CausalSelfAttention.forward(x, position_offset=0, cache=None, layer_idx=0)`, `Block.forward(x, position_offset=0, cache=None, layer_idx=0)`, `MinimalTransformerLM.forward(input_ids, cache=None)` — the `cache` parameter on `MinimalTransformerLM.forward` is consumed directly by Task 9's `generate.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transformer.py`:

```python
from llmtrain.model.cache import KVCache


def test_cached_decoding_matches_uncached_forward():
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=6, dropout=0.0
    )
    model = MinimalTransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, 16, (1, 5))

    with torch.no_grad():
        full_logits = model(input_ids)

    cache = KVCache()
    cached_logits = []
    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            step_logits = model(input_ids[:, t : t + 1], cache=cache)
            cached_logits.append(step_logits)
    cached_logits = torch.cat(cached_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transformer.py::test_cached_decoding_matches_uncached_forward -v`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'cache'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llmtrain/model/transformer.py`, update `CausalSelfAttention.forward`:

```python
    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        cache: "KVCache | None" = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x)
        k, v = kv.split(self.n_kv_heads * self.head_dim, dim=2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rotary_cos_sin(
            seq_len, self.head_dim, self.rope_theta, position_offset, x.device, x.dtype
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0, enable_gqa=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)
```

Add the import at the top of the file (for the type hint):

```python
from llmtrain.model.cache import KVCache
```

(Replace the string-literal type hint `"KVCache | None"` above with a plain `KVCache | None` now that it's imported.)

Update `Block.forward`:

```python
def forward(
    self,
    x: torch.Tensor,
    position_offset: int = 0,
    cache: KVCache | None = None,
    layer_idx: int = 0,
) -> torch.Tensor:
    x = x + self.attn(
        self.ln1(x), position_offset=position_offset, cache=cache, layer_idx=layer_idx
    )
    x = x + self.mlp(self.ln2(x))
    return x
```

Update `MinimalTransformerLM.forward`:

```python
    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = self.token_emb(input_ids)
        position_offset = cache.seq_len if cache is not None else 0
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, position_offset=position_offset, cache=cache, layer_idx=layer_idx)
        x = self.ln_f(x)
        return self.head(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transformer.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/transformer.py tests/test_transformer.py
git commit -m "feat: wire KVCache through attention and model forward passes"
```

---

## Task 9: `generate.py` core generation logic

**Files:**
- Create: `src/llmtrain/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `MinimalTransformerLM.forward(input_ids, cache=None)` (Task 8), `KVCache` (Task 7), `tokenizers.Tokenizer.encode`/`.decode`.
- Produces: `generate_token_ids(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0) -> list[int]` and `generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0) -> str` — `generate` is consumed by Task 10's CLI `main()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate.py`:

```python
import torch
from tokenizers import Tokenizer

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.generate import generate, generate_token_ids
from llmtrain.model.transformer import MinimalTransformerLM
from llmtrain.training.config import ModelConfig


def _tiny_setup() -> tuple[MinimalTransformerLM, Tokenizer]:
    tokenizer = train_tokenizer(["hello world", "hello there", "world hello there"], vocab_size=32)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=8,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        dropout=0.0,
    )
    model = MinimalTransformerLM(config)
    return model, tokenizer


def test_generate_token_ids_produces_requested_number_of_new_tokens():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    prompt_ids = tokenizer.encode("hello").ids
    output_ids = generate_token_ids(model, tokenizer, "hello", max_new_tokens=5, temperature=0.0)
    assert len(output_ids) == len(prompt_ids) + 5


def test_greedy_decoding_is_deterministic():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    output_a = generate(model, tokenizer, "hello", max_new_tokens=5, temperature=0.0)
    output_b = generate(model, tokenizer, "hello", max_new_tokens=5, temperature=0.0)
    assert output_a == output_b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.generate'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/llmtrain/generate.py`:

```python
import torch
from tokenizers import Tokenizer

from llmtrain.model.cache import KVCache
from llmtrain.model.transformer import MinimalTransformerLM


def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_token_ids(
    model: MinimalTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> list[int]:
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    cache = KVCache()
    generated_ids = list(prompt_ids)
    with torch.no_grad():
        logits = model(input_ids, cache=cache)
        next_id = _sample(logits[:, -1, :], temperature)
        generated_ids.append(next_id)
        for _ in range(max_new_tokens - 1):
            step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(step_input, cache=cache)
            next_id = _sample(logits[:, -1, :], temperature)
            generated_ids.append(next_id)

    return generated_ids


def generate(
    model: MinimalTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> str:
    token_ids = generate_token_ids(model, tokenizer, prompt, max_new_tokens, temperature)
    return tokenizer.decode(token_ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/generate.py tests/test_generate.py
git commit -m "feat: add KV-cache-backed text generation"
```

---

## Task 10: Persist tokenizer alongside checkpoints; `generate.py` CLI

**Files:**
- Modify: `src/llmtrain/training/train.py`
- Modify: `src/llmtrain/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `generate` (Task 9), `llmtrain.training.checkpoint.save_checkpoint`/`load_checkpoint` (existing), `tokenizers.Tokenizer.save`/`.from_file` (existing library API).
- Produces: `train.py` writes `<checkpoint_dir>/tokenizer.json` on every run; `generate.py` gains a `main()` CLI entry point (`--checkpoint`, `--tokenizer-path`, `--prompt`, `--max-new-tokens`, `--temperature`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_generate.py`:

```python
from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint


def test_generate_works_after_checkpoint_and_tokenizer_round_trip(tmp_path):
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "step_1.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    save_checkpoint(checkpoint_path, model, optimizer, step=1)
    tokenizer.save(str(tokenizer_path))

    loaded_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    loaded_config = ModelConfig(
        vocab_size=loaded_tokenizer.get_vocab_size(),
        d_model=8,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        dropout=0.0,
    )
    loaded_model = MinimalTransformerLM(loaded_config)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=0.0)
    load_checkpoint(checkpoint_path, loaded_model, loaded_optimizer)

    output = generate(loaded_model, loaded_tokenizer, "hello", max_new_tokens=3, temperature=0.0)
    assert isinstance(output, str)
```

(`Tokenizer` is already imported in `tests/test_generate.py` from Task 9 — no new import needed.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py::test_generate_works_after_checkpoint_and_tokenizer_round_trip -v`
Expected: This test actually exercises only existing (Task 9) machinery plus already-existing `save_checkpoint`/`load_checkpoint`/`Tokenizer.save`/`Tokenizer.from_file` — it may already PASS at this point. If it does, that confirms the round-trip works; proceed to Step 3 to add the CLI wrapper and tokenizer persistence, which are the actual deliverables of this task.

- [ ] **Step 3: Write the implementation**

In `src/llmtrain/training/train.py`, after the existing line:

```python
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

add:

```python
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    logger.info("saved tokenizer to %s", checkpoint_dir / "tokenizer.json")
```

In `src/llmtrain/generate.py`, add the new imports to the top of the file (alongside the existing `torch`/`Tokenizer`/`KVCache`/`MinimalTransformerLM` imports) and a CLI entry point at the bottom:

```python
import argparse
from pathlib import Path

from llmtrain.training.checkpoint import load_checkpoint
from llmtrain.training.config import ModelConfig
```

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = (
        Path(args.tokenizer_path)
        if args.tokenizer_path
        else checkpoint_path.parent / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ModelConfig() defaults must match the training-time architecture. train.py's
    # CLI doesn't yet override architecture fields (only max_steps/batch_size/lr/
    # checkpoint_dir), so this holds today; a future config-rightsizing spec that
    # adds architecture CLI overrides to train.py must persist them for generate.py too.
    model_cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = MinimalTransformerLM(model_cfg)
    # load_checkpoint requires an optimizer arg; inference discards it.
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    load_checkpoint(checkpoint_path, model, dummy_optimizer)

    output = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
    print(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `uv run pytest tests/ -v`
Expected: PASS (full suite, including the new round-trip test)

Run: `uv run ruff check .` and `uv run mypy src/`
Expected: no errors (fixes any import-order/typing issues surfaced by moving the inline imports to module level)

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/training/train.py src/llmtrain/generate.py tests/test_generate.py
git commit -m "feat: persist tokenizer with checkpoints and add generate.py CLI"
```

---

## Task 11: Re-run the local smoke test on the new architecture

**Files:** none (validation-only task, per the spec's required breaking-change follow-up)

**Interfaces:** none — this task consumes the fully wired pipeline from Tasks 1–10 to confirm end-to-end health.

- [ ] **Step 1: Run the full test suite one more time**

Run: `uv run pytest tests/ -v`
Expected: PASS (full suite)

- [ ] **Step 2: Run the local smoke test**

Run (per `docs/smoke-test.md`):

```bash
uv run --env-file .env python -m llmtrain.training.train \
    --dataset tiny_shakespeare --max-steps 50 --batch-size 4
```

Expected: exits 0, no traceback.

- [ ] **Step 3: Check the four smoke-test success criteria**

```bash
uv run python -c "import json; [json.loads(l) for l in open('app.log')]"
ls checkpoints/
```

Confirm: `app.log` parses as JSONL with no exception; `checkpoints/` contains at least one `step_*.pt` file; the W&B run URL printed to stdout shows a decreasing loss curve over the 50 steps.

- [ ] **Step 4: Exercise `generate.py` against the resulting checkpoint**

```bash
uv run python -m llmtrain.generate \
    --checkpoint checkpoints/step_50.pt --prompt "To be" --max-new-tokens 20
```

Expected: prints generated text with no traceback (adjust the checkpoint filename to whatever step number was actually saved last).

- [ ] **Step 5: Record the result**

```bash
git commit --allow-empty -m "chore: confirm tiny_shakespeare smoke test passes on modernized architecture"
```
