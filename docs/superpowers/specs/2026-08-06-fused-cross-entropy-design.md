# Fused Cross-Entropy — Design

Date: 2026-08-06

## Scope

Third and final spec from this session's brainstorming (after `2026-08-06-pretraining-loop-hardening-design.md` and `2026-08-06-validation-loop-design.md`). Closes the "fused cross-entropy" reconsideration flagged by this session's earlier audit: `vocab_size` grew to 32,768 as part of the architecture-modernization work, so the full `[batch*seq, vocab]` logits tensor materialized during training now costs real memory (~4.3GB at current defaults) — a cost that didn't exist when CLAUDE.md's original deferral note ("solves memory pressure this config doesn't currently have") was written.

**In scope:** `TransformerLM.forward`'s interface for exposing pre-head hidden states, a Liger-Kernel-backed fused linear+cross-entropy loss path, a shared `compute_loss()` helper coordinating with Spec B's `evaluate()`, and the CUDA-only dependency/fallback story.

**Explicitly deferred / excluded:**

- MoE, sequence packing, `num_workers` tuning, gradient checkpointing — separate specs or explicitly rejected (see prior specs and the architecture-modernization audit this session started from).
- Re-verifying Liger Kernel's cross-entropy math against a reference implementation — trusted as a well-regarded, independently-tested upstream library, consistent with the original ask that recommendations be grounded in "authoritative, well-regarded sources."
- `generate.py` — inference needs real per-token logit values for sampling (top-k/top-p), so it keeps using the existing full-logits `forward()` path unconditionally. Not touched by this spec.

## Investigation: why this isn't a `next_token_loss`-only change

Checked Liger Kernel's actual API via context7 before designing anything, since "fuse the cross-entropy" sounds like it could be a local change to `next_token_loss` but isn't:

- `LigerFusedLinearCrossEntropyLoss.forward(lin_weight, hidden_states, target)` fuses the final linear projection _and_ the loss into one chunked Triton computation — the full `[B*S, vocab]` logits tensor is never materialized in the first place.
- But `TransformerLM.forward()` currently always computes `self.head(x)` and returns full logits before `next_token_loss` is ever called — the memory cost already happened by then. Wrapping `F.cross_entropy` in something fused after the fact captures none of the benefit; the model's forward interface itself has to change to expose pre-head hidden states for training to route around the full projection.
- Confirmed via context7: CUDA installs require PyTorch ≥2.1.2 and Triton ≥2.3.0 (project already pins `torch>=2.6`, satisfied); `ignore_index` is supported directly on the loss module, matching the existing `pad_id`-based masking in `next_token_loss`.

Given the added interface change and the CUDA-only, external-dependency nature of a real fused kernel, three approaches were considered and discussed with the user before committing to a direction: (a) Liger Kernel's Triton kernel — chosen; (b) a hand-rolled chunked linear+cross-entropy loop in pure PyTorch — no new dependency and MPS/CPU-portable, but slower and meaningfully trickier to get log-sum-exp-stable across chunks; (c) deferring the spec entirely, since current peak logits memory (~4.3GB) isn't a demonstrated bottleneck on a 40/80GB A100. (a) was chosen for the real memory/throughput win on the hardware this project actually trains on.

## Components

### 1. `TransformerLM.forward`: expose pre-head hidden states

```python
def forward(
    self, input_ids: torch.Tensor, cache: KVCache | None = None, return_hidden: bool = False,
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

Default (`return_hidden=False`) behavior is byte-for-byte unchanged — every existing caller (`generate.py`, the MPS/CPU training fallback) is unaffected. `return_hidden=True` is only ever passed from the new fused-loss path below.

### 2. Fused loss path

```python
def next_token_loss_fused(
    hidden: torch.Tensor, head_weight: torch.Tensor, input_ids: torch.Tensor, pad_id: int,
) -> torch.Tensor:
    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    shift_targets = input_ids[:, 1:].reshape(-1)
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=pad_id)
    return loss_fn(head_weight, shift_hidden, shift_targets)
```

`head_weight` is `model.token_emb.weight` (tied to `head.weight` per the architecture-modernization spec). Accessed through `model` even when `torch.compile`-wrapped — `OptimizedModule` proxies attribute access to the underlying module, but this is flagged to verify empirically during implementation (a smoke-test-level check, not assumed from documentation) since `torch.compile` attribute proxying has known edge cases.

### 3. Shared `compute_loss()` — coordinates with Spec B

```python
def compute_loss(
    model: torch.nn.Module, input_ids: torch.Tensor, pad_id: int, use_fused_ce: bool,
) -> torch.Tensor:
    if use_fused_ce:
        hidden = model(input_ids, return_hidden=True)
        head_weight = model.token_emb.weight
        return next_token_loss_fused(hidden, head_weight, input_ids, pad_id)
    logits = model(input_ids)
    return next_token_loss(logits, input_ids, pad_id)
```

Both the training step and `evaluate()` (introduced in `2026-08-06-validation-loop-design.md`) call this instead of duplicating the fused/non-fused branch in two places — a small implementation-time coordination point between the two specs, not a change to Spec B's already-written document.

### 4. Config + dependency

- `TrainConfig.use_fused_ce: bool = True` + `--use-fused-ce`/`--no-use-fused-ce` CLI flag, matching the existing `--compile`/`--use-amp` `BooleanOptionalAction` pattern.
- Actual dispatch is `use_fused_ce_effective = train_cfg.use_fused_ce and device.type == "cuda"` — same CUDA-gating precedent already used for `torch.compile` and TF32 matmul precision. MPS/CPU always uses the plain `next_token_loss` path regardless of the flag.
- `liger-kernel` added as an **optional** dependency group (not a core dependency), and imported lazily inside the CUDA-gated branch (`from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss` only executes when `use_fused_ce_effective` is `True`). Local Mac dev never needs it installed or importable; the RunPod A100 setup step (`pip install wandb && wandb login`, per CLAUDE.md's RunPod workflow) gains an equivalent `uv sync --extra cuda` (or `pip install liger-kernel`) alongside it.

## Error handling

No new error paths beyond what a missing/failed lazy import already surfaces naturally (an `ImportError` if `use_fused_ce_effective=True` but `liger-kernel` isn't installed on a CUDA box — a clear, actionable failure rather than a silent fallback, consistent with the project's fail-fast convention). No validation added for `use_fused_ce` itself — trusted CLI/config input, matching existing convention.

## Testing strategy

CPU-only, tiny fake data, per CLAUDE.md's existing testing strategy:

- **Interface correctness (load-bearing)**: `model.head(model(input_ids, return_hidden=True))` produces the same output (within float tolerance) as `model(input_ids)` directly — confirms the `return_hidden` change is mathematically inert and the hidden-state/head split is wired correctly, independent of whether Liger Kernel is even installed.
- `compute_loss()` with `use_fused_ce=False` produces the same loss as calling `next_token_loss(model(input_ids), ...)` directly — confirms the shared helper's non-fused branch is a pure passthrough.
- The actual fused path (`use_fused_ce_effective=True`, real Liger Kernel call) cannot be exercised in CI or on local Mac dev (no CUDA) — validated by the manual smoke test on the real A100 run instead, the same convention already applied to `train()`/`main()` orchestration and noted as a currently-open follow-up (`docs/smoke-test.md` was deleted this session) in the pretraining-loop-hardening spec.

## Config changes summary

`TrainConfig` gains `use_fused_ce: bool = True`. `TransformerLM.forward` gains `return_hidden: bool = False` (default-inert). New `next_token_loss_fused()` and `compute_loss()` in `train.py`; `evaluate()` (Spec B) and the training step both route through `compute_loss()`. `liger-kernel` added as an optional dependency group in `pyproject.toml`, imported lazily. No `ModelConfig` changes.
