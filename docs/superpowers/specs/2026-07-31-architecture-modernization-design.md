# Architecture Modernization — Design

Date: 2026-07-31

## Scope

Bring `model/transformer.py` in line with modern small-LM practice while keeping the project's minimal, hand-rolled, single-purpose character (CLAUDE.md's overengineering guidance). This is the first of several independent follow-on specs identified while reviewing a batch of architecture/scaling suggestions; the others (config/scale rightsizing for the A100 fineweb-edu budget, training-loop robustness — LR schedule, weight decay, gradient accumulation — and data pipeline efficiency/packing) are deferred to their own specs.

**In scope:** RMSNorm, RoPE, SwiGLU MLP, weight tying, GQA, KV cache + a minimal `generate.py`.

**Explicitly deferred / excluded:**
- MoE — rejected outright; overkill for a dense toy-scale model.
- Fused cross-entropy / Liger kernel, elaborate VRAM budgeting, `torch.backends.cuda.enable_flash_sdp`-style global backend flags — not worth the complexity or are superseded APIs at this project's scale (see verification notes below).
- Config/scale rightsizing (vocab size, d_model/n_layers target for a 4–12hr A100 budget), LR schedule/weight decay/grad accumulation, sequence packing/pre-tokenization — separate specs.

**Breaking change:** this is not checkpoint-compatible with the current architecture (learned `pos_emb`, `LayerNorm`, GELU MLP). No migration path — only smoke-test checkpoints exist so far, no real pretraining run has happened yet. After implementation, re-run the existing local smoke test (`docs/smoke-test.md`) to reconfirm the pipeline end-to-end on the new architecture.

## Verification against current PyTorch docs (torch>=2.6, checked against 2.12 docs)

Per CLAUDE.md's "verify against current docs, don't assume" precedent (already established for the MPS/CUDA device-handling section):

- `torch.nn.RMSNorm` is a native built-in module (added 2.4) — no need to hand-roll RMSNorm.
- The current API for selecting an SDPA backend is the `torch.nn.attention.sdpa_kernel()` context manager with the `SDPBackend` enum, not the older `torch.backends.cuda.enable_flash_sdp()` global flags Gemini's suggestions used. This spec doesn't force a backend at all (letting SDPA auto-select, as the current code already does) — noted here only to avoid introducing the stale API if backend control is ever needed.
- `F.scaled_dot_product_attention(..., enable_gqa=True)` natively broadcasts fewer KV heads to match query heads (constraint: `n_heads % n_kv_heads == 0`) — no manual `repeat_interleave` needed in the common case.
- `is_causal=True` forms a causal bias for non-square (query/key length mismatched) matrices, but that bias is **top-left-aligned**, not tail-aligned — verified against `torch.nn.attention.bias.CausalBias` docs and reproduced independently (including with `enable_gqa=False`, ruling out a GQA-specific interaction). It does NOT treat the query as the tail of the full sequence, so it is unsafe for a KV-cache decode step with query length > 1 against a non-empty cache. The square case (training, uncached forward, prefill into an empty cache) is unaffected. The actual fix implemented is `is_causal=(seq_len > 1)`: a single-token cached decode step needs no causal mask at all (every cached key is a past position), so the non-square case that would be wrong is simply never asked to run.
- GQA is documented as "experimental," with support described for "FlashAttention and math kernel on CUDA tensors" — MPS/CPU support is unconfirmed. Treated as a risk below, not assumed to work.

## Components

### 1. RMSNorm
Replace `nn.LayerNorm(config.d_model)` with `torch.nn.RMSNorm(config.d_model)` at all three call sites (`Block.ln1`, `Block.ln2`, `MinimalTransformerLM.ln_f`). One-line-per-site change.

### 2. RoPE
- Remove `pos_emb` (`nn.Embedding(max_seq_len, d_model)`) and the additive positional term in `MinimalTransformerLM.forward`.
- Add a `rotary embedding` helper computing cos/sin **on the fly per forward call**, sized to the actual `seq_len` passed in — not precomputed to a fixed `max_seq_len` cache. This is what actually removes the hard context-length ceiling a learned embedding table bakes in; the cost is trivial next to attention itself.
- Applied to Q and K inside `CausalSelfAttention`, after the head split, before `F.scaled_dot_product_attention`.
- `ModelConfig` gains `rope_theta: float = 10000.0`.
- New validation: `head_dim % 2 == 0` (rotation operates on dimension pairs), alongside the existing `d_model % n_heads == 0` check.

### 3. SwiGLU MLP
Replace the GELU `nn.Sequential` with gated SiLU: `SiLU(w_gate(x)) * w_up(x)`, then `w_down`, then dropout. `d_ff = int(2/3 * 4 * d_model)` (standard formula keeping param count roughly aligned with the old 4× GELU MLP) — no extra rounding/multiple-of-N logic; not worth the config surface at this scale.

### 4. Weight tying
`self.head.weight = self.token_emb.weight` in `MinimalTransformerLM.__init__`, after both are constructed. `head` already has `bias=False`, so this is a direct assignment with no shape mismatch.

### 5. GQA (Grouped Query Attention)
- `ModelConfig` gains `n_kv_heads: int = 1` — an explicit static field (consistent with the dataclass's existing style, not auto-derived from `n_heads`); at the current default `n_heads=4` this is a 4:1 ratio. Must be set explicitly alongside `n_heads` in any future config resizing.
- New validation: `n_heads % n_kv_heads == 0`.
- `CausalSelfAttention` splits its combined `qkv_proj` into `q_proj` (`d_model → n_heads * head_dim`) and `kv_proj` (`d_model → 2 * n_kv_heads * head_dim`, split via `.split()`). Q reshapes to `(B, n_heads, S, head_dim)`; K/V reshape to `(B, n_kv_heads, S, head_dim)`.
- SDPA call adds `enable_gqa=True`.
- **Risk, from verification above**: if `enable_gqa=True` errors or misbehaves on CPU/MPS (the local smoke-test device), fall back to manually expanding K/V via `repeat_interleave(n_heads // n_kv_heads, dim=1)` before SDPA — always correct, just skips the memory optimization off-CUDA. Mirrors the existing CUDA-only gate pattern already used for `torch.compile` in `TrainConfig`. This will surface immediately in the CPU-only unit tests (fail-fast TDD), before any A100 time is spent.

### 6. KV Cache + `generate.py`
The codebase currently has no inference/generation path at all — a KV cache only has value paired with a loop that reuses it.

- New lightweight `KVCache`: per-layer `(k, v)` tensors; `update(layer_idx, k, v) -> (k_full, v_full)` concatenates new k/v onto cached ones along the sequence dimension.
- `CausalSelfAttention.forward` and `MinimalTransformerLM.forward` gain optional `cache: KVCache | None`, `layer_idx: int`, `position_offset: int` parameters.
- RoPE must rotate by **absolute** position (`position_offset + local_position`), not restart at 0 on every decode step — the load-bearing correctness detail for cached generation. `position_offset = cache.seq_len if cache else 0`, computed once in `MinimalTransformerLM.forward` and threaded through each block.
- New `generate.py`: loads a checkpoint + tokenizer, runs a prefill forward pass on the prompt with an empty cache, then loops single-token decode steps (greedy or temperature sampling) appending to the cache, until `max_new_tokens` or EOS. Thin CLI: `--checkpoint`, `--prompt`, `--max-new-tokens`, `--temperature`.

## Error handling

Two validation checks at `MinimalTransformerLM`/`CausalSelfAttention.__init__`, both raising `ValueError`, matching existing style:
- `d_model % n_heads == 0` (existing)
- `head_dim % 2 == 0` (new, for RoPE)
- `n_heads % n_kv_heads == 0` (new, for GQA)

No new error paths at forward time beyond what GQA's CPU/MPS fallback (above) already handles.

## Testing strategy

CPU-only, tiny fake data, per CLAUDE.md's existing testing strategy — extending `tests/test_transformer.py`:

- Existing shape/gradient tests continue to run against the new architecture (tiny config already has even `head_dim`).
- Weight tying: `model.head.weight is model.token_emb.weight`, and that it still receives a gradient.
- RoPE: the rotation helper at position 0 is the identity transform (`cos(0)=1`, `sin(0)=0`) — a direct unit test of the helper independent of the full model.
- No positional ceiling: forward pass succeeds for a `seq_len` larger than any previously-used `max_seq_len`, locking in that RoPE removed the hard bound.
- GQA: shape test confirming K/V head count is `n_kv_heads` not `n_heads`; a CPU/MPS `enable_gqa` compatibility check (exercises the fallback path if needed).
- KV cache correctness (load-bearing): a full uncached forward pass over a sequence vs. token-by-token cached decode must produce matching logits (within float tolerance) at each position.
- `generate.py`: smoke-level test (tiny model, tiny vocab, fixed seed) asserting it produces `max_new_tokens` tokens without error, and that greedy decoding is deterministic across two runs.

## Config changes summary

`ModelConfig` gains: `rope_theta: float = 10000.0`, `n_kv_heads: int = 1`. `pos_emb`-related nothing removed from config (`max_seq_len` stays — still used for tokenizer truncation and dataset collation, just no longer bounds an embedding table).
