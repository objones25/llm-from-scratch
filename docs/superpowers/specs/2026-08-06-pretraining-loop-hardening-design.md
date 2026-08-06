# Pretraining-Loop Hardening — Design

Date: 2026-08-06

## Scope

This is the "training-loop robustness" follow-on spec deferred by `2026-07-31-architecture-modernization-design.md` (LR schedule/weight decay/gradient accumulation). LR warmup+cosine decay already shipped separately (see git history); this spec covers what's left: gradient accumulation, gradient clipping, AdamW hyperparameter retuning, and weight initialization — the pieces standing between the current `train()` loop and a training recipe that behaves like real LLM pretraining, not just a working forward/backward loop.

Second of three planned specs from the current brainstorming session (this one, then held-out validation loop, then fused cross-entropy — each independent, each getting its own spec).

**In scope:** `TrainConfig.gradient_accumulation_steps`, `TrainConfig.grad_clip`, `TrainConfig.beta2`/`weight_decay` retune, `TransformerLM._init_weights`, bias removal from all `nn.Linear` layers.

**Explicitly deferred / excluded:**

- Held-out validation loop, sequence packing, `num_workers` tuning, fused cross-entropy, MoE, gradient checkpointing — separate specs or explicitly rejected (see prior spec and the architecture-modernization audit this session started from).
- `warmup_steps` default value: analysis below shows it doesn't need to change once "step" is redefined to mean optimizer-step.
- Backward compatibility with old smoke-test docs/checkpoints — both were deleted by the user during this session; no migration path needed.

## Verification against current PyTorch docs

Per CLAUDE.md's "verify against current docs, don't assume" precedent, checked via context7 against current PyTorch docs:

- `torch.nn.utils.clip_grad_norm_(parameters, max_norm)` computes the pre-clip total norm (L2 by default), scales all gradients in-place by `min(max_norm / (total_norm + 1e-6), 1.0)` (never amplifies), and **returns the pre-clip total norm** — this is what gets logged to W&B as `grad_norm`.
- The `scaler.unscale_(optimizer)`-before-clip pattern in PyTorch's AMP docs applies only when a `GradScaler` is in use (float16 training). This project uses `bfloat16` on CUDA with no `GradScaler` (confirmed: no `GradScaler` usage anywhere in the codebase) — gradients are never scaled in the first place, so `clip_grad_norm_` can be called directly on `model.parameters()` with no unscale step.
- PyTorch's own gradient-accumulation-under-AMP guidance: the optimizer step (and, if a scaler were in use, its update) should only happen once per full effective batch, at effective-batch granularity — confirms clipping and `optimizer.step()` belong at the end of the accumulation window, not per micro-batch.
- Idiomatic custom weight init is `model.apply(fn)` with an `isinstance` dispatch inside `fn`, and PyTorch's own example wraps `fn` in `@torch.no_grad()` to avoid tracking the initialization ops in the autograd graph. `nn.Linear`'s default `reset_parameters()` uses `kaiming_uniform_(a=sqrt(5))` for weights and a fan-in-scaled uniform for bias — confirms what `_init_weights` is overriding away from, and that dropping `bias=False` removes the bias branch entirely (no bias to initialize).

## Components

### 1. Gradient accumulation

- New `TrainConfig.gradient_accumulation_steps: int = 8` + `--gradient-accumulation-steps` CLI flag (default reads from the dataclass field, per existing convention).
- **Step semantics change**: `step` becomes an optimizer step (post-accumulation), not a micro-batch forward/backward. `get_lr()`, `--max-steps`, `checkpoint_interval`, and `wandb.log(..., step=step)` keep their current signatures and meaning unchanged.
- At current defaults (`batch_size=32`, `max_seq_len=2048`, `gradient_accumulation_steps=8`, `max_steps=10000`), this yields effective batches of `32 × 8 × 2048 = 524,288` tokens/step and `5.24B` total tokens — deliberately above the ~1.5B Chinchilla-optimal point for this model's ~75.5M non-embedding params, matching the "overtrain a small model for inference quality" choice LLaMA makes. No change to `max_steps` needed.
- `warmup_steps` stays at its current default (`200`). Under the new step semantics this is `200 × 524,288 ≈ 105M` warmup tokens, ~2% of the total run — a normal warmup fraction. (Previously, under micro-batch-step semantics, this same constant meant `200 × 65,536 ≈ 13M` tokens, ~2% of the _old_ 0.655B-token budget — the ratio was already fine; only the absolute token budget was wrong, and accumulation fixes that directly.)
- Implementation: an inner micro-batch counter (module-level to the accumulation window, independent of the outer `while step < max_steps: for batch in dataloader:` epoch-restart structure) tracks position within the current window. A window spanning an epoch boundary (relevant for small datasets like `tiny_shakespeare`; a no-op for streaming datasets that never exhaust) simply keeps accumulating rather than resetting — no special-casing required.
- Loss is divided by `gradient_accumulation_steps` before `.backward()`. `optimizer.step()`, `optimizer.zero_grad()`, `step += 1`, and checkpoint saving only fire once per full window — checkpoints always land on a clean accumulation boundary, so `--resume` never needs to reconstruct partial-window state.
- W&B logs the mean loss over the completed window once per optimizer step (not once per micro-batch), keeping `loss`/`lr`/`grad_norm` on the same step axis.

### 2. Gradient clipping + grad-norm logging

- New `TrainConfig.grad_clip: float = 1.0` + `--grad-clip` CLI flag.
- `total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)` called once per accumulation window, after all micro-batch backward passes and before `optimizer.step()`.
- `total_norm.item()` logged to W&B as `grad_norm` alongside `loss`/`lr` — closes the "no grad norm observability" gap flagged by this session's earlier audit, at no extra cost since `clip_grad_norm_` computes it regardless.

### 3. AdamW retune

- `TrainConfig.beta2`: `0.999` → `0.95`. PyTorch's `AdamW` default (`0.999`) is a known source of loss spikes at LLM pretraining scale; `0.95` matches GPT-3/LLaMA-family practice.
- `TrainConfig.weight_decay`: `0.01` → `0.1`. Current default is 10x weaker than standard LLM pretraining practice.
- `beta1` (`0.9`) is already correct, unchanged.

### 4. Weight initialization + bias removal

- All `nn.Linear` layers in `CausalSelfAttention` (`q_proj`, `kv_proj`, `out_proj`) and `MLP` (`w_gate`, `w_up`, `w_down`) gain `bias=False`, matching LLaMA/PaLM/GPT-NeoX convention. No bias parameters remain anywhere in the model (`nn.RMSNorm`'s learnable gain is a scale, not a bias, and is left untouched).
- New `_init_weights(module)` function in `transformer.py`, wrapped in `@torch.no_grad()`:
  - `nn.Linear` and `nn.Embedding` weights: `nn.init.normal_(weight, mean=0.0, std=0.02)`.
  - Residual-stream output projections specifically (`attn.out_proj`, `mlp.w_down`) additionally scaled by `1/√(2·n_layers)` after the above (GPT-2/nanoGPT convention — controls residual-stream variance growth with depth; these are the two places per layer that write directly into the residual stream).
- Applied via `self.apply(self._init_weights)` in `TransformerLM.__init__`, called _after_ weight tying (`self.head.weight = self.token_emb.weight`). `apply()` visits both the `token_emb` and `head` submodules separately, so the shared tensor gets `normal_`-initialized twice (once per module visit) — harmless since both draws come from the identical `N(0, 0.02²)` distribution and the tensor stays a single shared object throughout (whichever draw happens last is what sticks for both), but worth noting explicitly so it doesn't read as a bug during review.

## Error handling

No new error paths. Existing validation (`d_model % n_heads == 0`, `n_heads % n_kv_heads == 0`, `head_dim % 2 == 0`) is unaffected. `gradient_accumulation_steps` and `grad_clip` are trusted CLI/config inputs, consistent with the project's existing convention of not validating internal/trusted config values.

## Testing strategy

CPU-only, tiny fake data, per CLAUDE.md's existing testing strategy:

- **Gradient accumulation correctness (load-bearing)**: on a tiny toy model, accumulating gradients over N micro-batches (loss divided by N, N `.backward()` calls, one `optimizer.step()`) must produce gradients equal (within float tolerance) to a single forward/backward over the full concatenated batch. Pure, deterministic, no GPU/network required.
- **Weight init**: sample `TransformerLM.token_emb.weight` (a large-enough tensor) and assert its empirical std is close to `0.02`; assert no `nn.Linear` submodule has a non-`None` `.bias`.
- **Residual-projection scaling**: assert `out_proj`/`w_down` weight std is scaled down relative to a plain `std=0.02` init by the expected `1/√(2·n_layers)` factor, for a config with `n_layers > 1`.
- **Gradient clipping**: on a tiny model, force artificially large gradients, call `clip_grad_norm_` at `train_cfg.grad_clip`, assert the resulting post-clip norm is ≤ `grad_clip` (within tolerance) and that the returned pre-clip norm matches a manually computed L2 norm.
- `train()`/`main()` orchestration itself remains untested by design (per existing convention) — validated by manual smoke test instead. **Note:** `docs/smoke-test.md` was deleted this session; whether to write a replacement is an open follow-up, not blocking this spec.

## Config changes summary

`TrainConfig` gains: `gradient_accumulation_steps: int = 8`, `grad_clip: float = 1.0`. `TrainConfig.beta2` changes `0.999` → `0.95`; `TrainConfig.weight_decay` changes `0.01` → `0.1`. Both new fields get corresponding `--gradient-accumulation-steps`/`--grad-clip` CLI flags in `main()`, following the existing "default reads from the dataclass field" pattern. No `ModelConfig` field changes — bias removal and weight init are internal to `transformer.py`'s module construction, not config surface.
