# SFT Support — Design

Date: 2026-08-08

## Scope

Follows the first real pretraining checkpoints (`step_10000.pt` on `fineweb_edu`, generating coherent-but-imperfect text per CLAUDE.md's dataset workflow: `tiny_shakespeare` → `reformer_enwik8` → `fineweb_edu` → `smoltalk` SFT). This spec adds the pieces needed to run that last stage.

**In scope:**

- A weights-only checkpoint init path (`--init-from-checkpoint`) distinct from `--resume`, for starting a fresh training run (fresh step counter, fresh dataset stream, no optimizer state) from pretrained weights.
- Chat-formatted data support (`smoltalk`, `no_robots`) with prompt/response loss masking, so the loss is only computed on assistant-authored tokens.
- Reusing the existing `[PAD]` token as an end-of-turn/stop signal, so the model learns where a response ends without any vocabulary change.
- A small `generate.py` change so decoding actually stops on that signal.

**Explicitly deferred / excluded:**

- New special vocab tokens or embedding resizing — precisely what reusing `[PAD]` avoids. Any future EOS token would need embedding-table expansion and is out of scope here.
- Fixing `--resume`'s implicit tokenizer-reproducibility assumption (it retrains the tokenizer from the same sample/seed rather than persisting it) — pretraining-only, untouched by this work. `--init-from-checkpoint` sidesteps the issue entirely by loading the tokenizer from disk instead.
- New `TrainConfig` dataclass fields for SFT hyperparameters — lr/min-lr/warmup/max-steps are already CLI flags; SFT just gets invoked with different values.
- Whole-string offset-mapped tokenization of chat turns — turn-by-turn tokenize-and-concatenate is the accepted simpler approach (see Components §1).
- Sequence packing, `DataLoader` worker tuning, gradient checkpointing — already deferred per CLAUDE.md, unrelated to this work.

## Investigation: chat dataset schemas

Checked directly against the Hugging Face Hub rather than assumed:

| Dataset                   | Config    | Splits                         | Schema                                                               |
| ------------------------- | --------- | ------------------------------ | -------------------------------------------------------------------- |
| `HuggingFaceTB/smoltalk`  | `all`     | `train` (1.0M), `test` (54.9K) | `messages: list[{role, content}]`, `source`                          |
| `HuggingFaceH4/no_robots` | `default` | `train` (9.5K), `test` (500)   | `prompt`, `prompt_id`, `messages: list[{role, content}]`, `category` |

Both expose the same `messages` shape and a native `train`/`test` split — no holdout-carving needed, unlike `reformer_enwik8`/`fineweb_edu` in the validation-loop design. `smoltalk`'s `all` config is the merged full corpus (2.2M rows total across all subsets); `no_robots` is small enough to serve as `smoltalk`'s fast local-smoke-test counterpart, the same role `tiny_shakespeare` plays for `fineweb_edu`.

## Key design decisions

1. **`[PAD]` as the stop signal**, not a new EOS token. A new special token changes `vocab_size`, and the model's `token_emb`/`head` weights (weight-tied) are shaped by the exact vocab the checkpoint was trained with — adding a token would break `load_state_dict` on every existing checkpoint (including the real `step_10000.pt` run this project already has). `[PAD]` is repurposed instead: supervised (real target) immediately after each assistant turn, still ignored everywhere else it appears as filler. No vocab change, no embedding resize, fully backward-compatible with existing checkpoints.
2. **ChatML-style tags** (`<|user|>`, `<|assistant|>`, etc.) formatted as plain text through the existing frozen byte-level BPE tokenizer — these are literal characters tokenized like any other text, not new vocab entries.
3. **All assistant turns supervised**, not just the final one — standard SFT practice, makes full use of multi-turn examples in `smoltalk`.
4. **Both `smoltalk` and `no_robots` wired into the registry** — mirrors the existing fast-local/full-scale pairing pattern (`tiny_shakespeare`/`fineweb_edu`).
5. **Model architecture auto-adopted from the checkpoint's persisted `model_config`** on `--init-from-checkpoint`, the same pattern `generate.py` already uses — makes a shape-mismatch load impossible and avoids duplicating architecture info the checkpoint already has.

## Components

### 1. `src/llmtrain/data/chat.py` (new module)

```python
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
            labels.extend(turn_ids)       # supervised
            input_ids.append(pad_id)      # stop signal
            labels.append(pad_id)         # ...and it's supervised too
        else:
            labels.extend([IGNORE_INDEX] * len(turn_ids))
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    pad_amount = max_seq_len - len(input_ids)
    input_ids.extend([pad_id] * pad_amount)
    labels.extend([IGNORE_INDEX] * pad_amount)   # tail filler: never supervised
    return input_ids, labels

def encode_chat_batch(
    tokenizer: Tokenizer, examples: list[dict], pad_id: int, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [encode_chat_example(tokenizer, ex["messages"], pad_id, max_seq_len) for ex in examples]
    input_ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    labels = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return input_ids, labels
```

Turn-by-turn tokenize-and-concatenate (not one `tokenizer.encode()` call over the whole formatted string with offset-mapping) — simpler code, exact span alignment by construction, at the cost of very rare BPE boundary mismatches at turn seams versus what whole-string tokenization would produce. Newline-delimited ChatML tags make this practically a non-issue for a toy-scale project. Truncation (`[:max_seq_len]`) can, in rare long-conversation cases, cut off mid-turn or drop the trailing stop-signal token entirely — accepted as a known limitation, not handled specially.

### 2. `src/llmtrain/data/streaming.py`

`DatasetSpec` gains one field:

```python
@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    split: str
    text_column: str = "text"
    val_split: str | None = None
    val_holdout_examples: int | None = None
    messages_column: str | None = None   # new
```

When set, the dataset is chat-formatted: `load_streaming_datasets` skips the `text_column` rename for it entirely (chat examples keep `messages` — and any other columns — untouched). New registry entries:

```python
"smoltalk": DatasetSpec(
    path="HuggingFaceTB/smoltalk", name="all", split="train",
    messages_column="messages", val_split="test",
),
"no_robots": DatasetSpec(
    path="HuggingFaceH4/no_robots", name="default", split="train",
    messages_column="messages", val_split="test",
),
```

No changes to `load_streaming_datasets`'s control flow beyond guarding the rename (`if spec.text_column != "text" and spec.messages_column is None`) — the native-`val_split` branch already used by `tiny_shakespeare` covers both new entries as-is.

### 3. `src/llmtrain/training/train.py`

**Unified `(input_ids, labels)` loss interface.** `IGNORE_INDEX = -100` module constant. `next_token_loss`/`next_token_loss_fused` shift and compare against `labels` instead of `input_ids`:

```python
def next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
```

`compute_loss(model, input_ids, labels, use_fused_ce)` takes both tensors; `next_token_loss_fused` similarly shifts `labels` instead of `input_ids`. `evaluate()` drops its `pad_id` parameter entirely (masking now lives in `labels`, not a loss-time kwarg) and, like the training loop, unpacks `(input_ids, labels)` from each batch and moves both to device.

**Dataset-kind-aware collate.** `make_collate_fn(tokenizer, max_seq_len, messages_column)`:

```python
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

Pretraining behavior is unchanged in effect — `labels` masks pad positions exactly like the old `ignore_index=pad_id` did, just constructed at collate time instead of passed as a loss kwarg.

**Weights-only init.** New mutually exclusive CLI flags `--init-from-checkpoint PATH` / `--tokenizer-path PATH`, alongside the existing `--resume PATH` (enforced via an `argparse` mutually exclusive group). `--dataset` choices extended with `"smoltalk"`, `"no_robots"`. When `--init-from-checkpoint` is given, `train()`:

- Resolves the checkpoint and tokenizer paths via `s3.py`'s `resolve_local_path`/`sibling_path` (same helpers `generate.py` already uses), so `s3://` checkpoints work with no manual download — matters in practice, since the project's actual checkpoints live on a RunPod network volume exposed over S3.
- Loads the tokenizer with `Tokenizer.from_file(...)` instead of calling `train_tokenizer(...)` — the SFT run must use the exact tokenizer the pretrained embeddings were trained with, not a freshly retrained one over `smoltalk`/`no_robots` text.
- Peeks the checkpoint's persisted `model_config` (same as `generate.py`) and builds `ModelConfig` from it, overriding any CLI architecture flags; logs a warning if a CLI flag was explicitly passed and disagrees with the checkpoint's value.
- Calls `load_checkpoint(path, model, optimizer=None)` — model weights only. The returned `step`/`dataset_state` are discarded: the SFT run starts at `step = 0` with a fresh stream over the SFT dataset, not the pretraining run's resumed position.

`--resume`'s existing behavior (tokenizer retraining, optimizer/step/dataset restore) is untouched — it still exists for resuming an interrupted run of the _same_ phase (pretraining or SFT), just now alongside the new flag rather than instead of it.

### 4. `src/llmtrain/generate.py`

`generate_token_ids`'s decode loop breaks before appending a sampled token that equals `pad_id`:

```python
next_id = sample_next(logits[:, -1, :])
if next_id == pad_id:
    break
generated_ids.append(next_id)
```

Requires threading `pad_id` (from the loaded tokenizer) into `generate_token_ids`. No-op for pure-pretraining checkpoints/tokenizers where `[PAD]` is never sampled during normal generation (it's never a supervised target in that setting, so the model has no incentive to produce it).

## Error handling

No new validation paths beyond the existing `argparse` mutually-exclusive-group enforcement for `--resume`/`--init-from-checkpoint` (fails at CLI-parse time with a clear message, before any model/dataset loading happens). A CLI architecture flag that disagrees with the checkpoint's persisted `model_config` is a logged warning, not an error — the checkpoint's value always wins, so there's no way for it to produce a shape mismatch.

## Testing strategy

CPU-only, tiny fake data, per CLAUDE.md's existing testing strategy:

- `data/chat.py`: `encode_chat_example` on a tiny fake tokenizer + 2-3 turn conversation — assert assistant-turn positions get real token-id labels, user/system-turn positions get `IGNORE_INDEX`, exactly one `pad_id` is spliced in (and supervised) after each assistant turn, and tail padding beyond the last real token is `IGNORE_INDEX`. A separate case for truncation: a conversation longer than `max_seq_len` is cut to exactly `max_seq_len` for both `input_ids` and `labels`.
- `make_collate_fn`: both branches (`messages_column=None` vs set) against fake examples, asserting the `(input_ids, labels)` shapes/masking match expectations.
- `next_token_loss`/`next_token_loss_fused`: updated to the new `labels`-based signature; assert `IGNORE_INDEX` positions don't contribute to the loss (e.g. compare loss with/without extra ignored positions appended).
- `load_checkpoint(path, model, optimizer=None)`: assert model weights load correctly and the call doesn't require or touch optimizer state (already close to existing coverage per `checkpoint.py`'s docstring comments — extend if not already covered).
- `generate_token_ids`: fake model that always emits `pad_id` as position N's argmax — assert generation stops at length N and doesn't append the pad token itself past that point.
- `DatasetSpec`/registry: `smoltalk` and `no_robots` entries have `messages_column="messages"`, `val_split="test"`, `val_holdout_examples=None`.
- `train()`/`main()` orchestration remains untested by design (per existing convention) — validated by a manual smoke test: `--init-from-checkpoint` against a small local pretraining checkpoint, `--dataset no_robots`, a handful of steps, confirming `val_loss` is finite/decreasing and `generate.py` against the resulting checkpoint stops before `max_new_tokens` at least some of the time.

## Config changes summary

`DatasetSpec` gains `messages_column: str | None = None`. Two new `DATASET_REGISTRY` entries (`smoltalk`, `no_robots`). `next_token_loss`/`next_token_loss_fused`/`compute_loss` signatures change from `(..., input_ids, pad_id)` to `(..., labels)` — breaking change to their call sites in `train.py`'s loop and `evaluate()`, updated as part of this spec. `make_collate_fn` gains a `messages_column` parameter and its return type changes from `torch.Tensor` to `tuple[torch.Tensor, torch.Tensor]`. New CLI flags on `train.py`: `--init-from-checkpoint`, `--tokenizer-path` (mutually exclusive with `--resume`). `generate_token_ids` gains pad-token stop behavior. No `TrainConfig`/`ModelConfig`/`GenerationConfig` dataclass field changes.
