# DPO Pipeline (RLAIF-style) — Design

Date: 2026-08-18

## Scope

Adds a post-SFT preference-alignment stage on top of the existing v2-scale SFT checkpoint
(431M non-embedding params, `d_model=1440`/`n_layers=20`, W&B run `0bcu0prs`, smoltalk-trained).
The pipeline generates preference pairs by sampling two completions per prompt from our own SFT
model and having an external LLM judge pick the better one (RLAIF), then trains on those pairs
with Direct Preference Optimization (DPO) via TRL's `DPOTrainer`.

**In scope:** a three-stage pipeline (`generate_pairs.py` → `judge.py` → `training/dpo.py`), an
HF-`transformers`-compatible wrapper around `TransformerLM` so `DPOTrainer` can be used unmodified
rather than reimplemented, and a judge design that specifically defends against position bias,
verbosity bias, and self-enhancement bias.

**Explicitly deferred / excluded:**

- **PPO.** Requires training a separate reward model plus a full RL loop (rollouts, value
  function, KL control) — meaningfully more infrastructure and instability risk than DPO for a
  personal project, without a proportionate payoff at this model's scale. PPO is generally
  more performant than DPO in the literature; that trade-off is a deliberate, acknowledged choice
  here, not an oversight.
- **Factuality decomposition scoring** (atomic-fact extraction + weighted aggregate, RAG/web-search
  fact-checking). Out of scope for both judge-call budget and engineering complexity; the judge
  still does a basic factuality/correctness pass as part of its ordinary pairwise comparison, just
  not the full decompose-and-verify pipeline.
- **PEFT/LoRA for the DPO stage.** The model is small enough (431M params) that a full fine-tune,
  including holding the frozen reference-model copy TRL creates automatically, comfortably fits
  A100 memory. No reason to add LoRA's complexity for this.
- **Dataset stream-resume for this pipeline's stages.** Unlike `fineweb_edu` pretraining, every
  stage here runs in minutes over a few thousand rows. A RunPod spot preemption means re-running
  the (short) stage from scratch, not implementing resume machinery sized for a multi-hour run.
- **A custom prompt dataset.** Uses `trl-lib/ultrafeedback-prompt` (37.8K train / 2K test,
  prompt-only, TRL-maintained) rather than building one from scratch.

## Motivation

The SFT checkpoint (`step_12000.pt`, val_loss 1.001) produces fluent, well-structured,
instruction-following output but is weak on factual grounding and multi-step reasoning — expected
for a 431M model trained on a fineweb-edu-scale corpus, confirmed by manual generation testing
(correct on "capital of France", garbled on "why is the sky blue", wrong on basic arithmetic).
DPO won't close that capacity/knowledge gap — it's a preference-alignment method, not a knowledge
source — but it can sharpen instruction-adherence, reduce confidently-wrong tone, and improve
style/helpfulness within what the model already knows. Expectations for this pipeline should stay
calibrated to that: better alignment on top of an unchanged knowledge ceiling.

## Architecture

New files, flat at `src/llmtrain/` top level (matching the existing `generate.py`/`s3.py`
convention — no nested feature folders):

```text
src/llmtrain/
  generate_pairs.py        # stage 1 CLI: prompts -> 2 sampled completions each
  judge.py                 # stage 2 CLI: completions -> judge calls -> filtered DPO pairs
  model/hf_wrapper.py       # PreTrainedModel/PretrainedConfig adapter around TransformerLM,
                            # + PreTrainedTokenizerFast wrap helper
  training/dpo.py           # stage 3 CLI: DPOConfig/DPOTrainer wiring, entry point
```

## Data flow

Each stage writes a plain JSONL artifact — inspectable, diffable, and re-runnable independently
(e.g. re-judging without re-generating if the judge prompt needs revision):

```text
trl-lib/ultrafeedback-prompt (HF Hub, streamed)
        |  generate_pairs.py  (samples 2 completions/prompt from our SFT checkpoint,
        |                      reuses generate.py's KV-cache generation code)
        v
pairs_raw.jsonl   { "prompt": "...", "completion_a": "...", "completion_b": "..." }
        |  judge.py  (2x judge calls per pair, swapped order; see Judge design below)
        v
pairs_dpo.jsonl   { "prompt": "...", "chosen": "...", "rejected": "..." }
        +  logged summary: total in, kept, discard counts by reason, length-ratio diagnostic
        |  training/dpo.py  (formats via data/chat.py's format_turn, loads via
        |                    datasets.load_dataset("json", ...), trains with DPOTrainer)
        v
DPO-tuned checkpoint, exported back through the existing save_checkpoint() format
```

`prompt`/`completion_a`/`completion_b`/`chosen`/`rejected` are stored as raw text (not
pre-formatted with chat tags) so the intermediate files stay human-readable and directly editable
— formatting via `format_turn()` happens once, at `training/dpo.py` load time.

`--output-dir` is a CLI flag on every stage (no hardcoded paths). These JSONL files are small
enough (thousands of short rows) that, unlike the multi-GB model checkpoints, no S3/network-volume
handling is needed for them.

## Judge design

**Access**: HF Inference Providers via `huggingface_hub.InferenceClient`, using the existing
`HF_TOKEN`. Default `provider="together"`, `model="meta-llama/Llama-3.3-70B-Instruct"` — chosen
over the initially-considered `Llama-3.1-70B-Instruct` because that model is currently live on
only `featherless-ai`, whose structured-output support isn't confirmed by HF's own docs, while
`Llama-3.3-70B-Instruct` is live on `together` (the provider HF's own `InferenceClient` docs use in
their canonical `response_format` example, and independently corroborated). Provider/model
availability changes over time, so this default is not treated as a permanent guarantee — see
"Startup self-check" below. `temperature=0.15` for reproducibility across runs.

Neither model choice creates self-enhancement-bias risk: the judge is an unrelated model family
from our own 431M `TransformerLM`, by construction, regardless of which is used.

**Structured output, not text parsing**: the judge call uses `response_format={"type":
"json_schema", ...}` with a schema requiring `{"reasoning": string, "verdict": "A"|"B"}` (enum
constraint, `strict: true`). This replaces an earlier draft that asked for free-text reasoning
followed by a `FINAL: A`/`FINAL: B` line parsed via regex — fragile if the model deviates from the
convention. `reasoning` is retained (for later inspection/debugging) but the verdict itself is
schema-enforced, not regex-extracted. Support for structured outputs is provider-dependent per
HF's own docs, which is exactly why the model/provider default above was chosen for its stronger
(if still not airtight) corroboration, and why the pipeline verifies it empirically rather than
trusting the claim (below).

**Prompt guidelines** (embedded in the judge's system/user prompt):

- Judge only correctness, helpfulness, and relevance to the prompt.
- Explicit instruction not to prefer a response merely for being longer/more detailed, plus one
  in-context example demonstrating a shorter correct response beating a padded one.
- No ties allowed — must pick exactly one response, breaking close calls on correctness, then
  relevance, then clarity, in that order (a DPO pair fundamentally needs a strict chosen/rejected
  order; a genuine tie contributes no gradient).
- Reasoning is requested before the verdict field (keeps the chain-of-thought effect on judge
  quality) even though it's schema-structured rather than free text.

**Position-bias mitigation**: every pair is judged twice — once as (A=completion_a,
B=completion_b), once swapped (A=completion_b, B=completion_a). Each verdict is mapped back to
which _original_ completion won.

- Both calls agree -> keep as `(chosen, rejected)`.
- Verdicts disagree -> discard, tagged `position_bias_disagreement`, and counted (not silently
  dropped) so a high discard rate is a visible signal that the judge prompt itself needs revision.
- A response that fails to parse as valid schema-conforming JSON, or omits/mis-values `verdict` ->
  discard, tagged `parse_failure` (kept as a separate counter from disagreement, since one means
  "the API/schema integration is broken" and the other means "the comparison was genuinely too
  close" — conflating them would hide which failure mode is actually occurring).

**Verbosity-bias monitoring**: for every kept pair, log the token-length ratio of chosen vs.
rejected. This is a diagnostic alongside the discard-rate summary, not a hard filter (a length-based
filter would just be an second, undeclared judge) — a systematically skewed ratio is the signal to
revise the prompt.

**Startup self-check**: before processing the full batch, `judge.py` makes one real judge call and
verifies it returns valid schema-conforming JSON. Fails fast with a clear error on the actual
response if not, rather than discovering a provider/schema incompatibility partway through a paid
multi-thousand-call batch. 1-2 fallback provider/model pairs are documented for this check to try
next (a config change, not a code change) if the default fails.

**Retry/error handling**: judge API calls get bounded retry-with-backoff on transient errors
(mirrors `checkpoint.py::save_checkpoint`'s existing retry pattern for network-volume writes),
then the pair is tagged `api_failure` and skipped rather than aborting the whole batch.

**Output**: `judge.py` writes `pairs_dpo.jsonl` plus a summary: total pairs in, kept, and discard
counts by reason (`position_bias_disagreement` / `parse_failure` / `api_failure`), plus the
length-ratio diagnostic.

## Generation stage (`generate_pairs.py`)

Loads prompts from `trl-lib/ultrafeedback-prompt` (train split), samples 2 independent completions
per prompt from the SFT checkpoint (`step_12000.pt`) via the existing KV-cache generation code in
`generate.py` (reused, not duplicated) — both at the same sampling settings (temperature/top-k/
top-p), so the two completions are a fair apples-to-apples comparison for the judge rather than
one greedy and one sampled. Target dataset size: ~2,000 kept preference pairs after stage 2's
filtering (small but standard for a toy/personal-scale DPO run — keeps both HF Inference API spend
and pod GPU time modest, and fast to iterate on the judge prompt before committing to a larger
batch).

## DPO training stage (`training/dpo.py`)

**`model/hf_wrapper.py`**:

- `TransformerLMConfig(PretrainedConfig)` mirrors `ModelConfig`'s fields (`d_model`, `n_layers`,
  `n_heads`, `n_kv_heads`, `dropout`, `rope_theta`, `vocab_size`); sets `tie_word_embeddings=False`
  since weight tying is already handled directly on the parameter in `TransformerLM.__init__`, so
  HF's own tie-weights machinery should stay inert rather than fight it.
- `TransformerLMForCausalLM(PreTrainedModel)` holds `self.model = TransformerLM(...)` as a
  submodule. `forward(input_ids, attention_mask=None, **kwargs)` calls `self.model(input_ids)` and
  returns `CausalLMOutputWithPast(logits=...)`. `attention_mask` is accepted (required by HF's
  calling convention) but **not passed through** — `TransformerLM.forward()` has no such
  parameter at all, relying purely on causal ordering. This is only correct if DPO's batches are
  right-padded, the same assumption `make_collate_fn`'s SFT collation already relies on (causal
  masking alone prevents attending into a right-padded tail). **Confirmed empirically**: a real
  `DPOTrainer` built against a tiny instance of this wrapper, batch pulled via
  `trainer.get_train_dataloader()`, right-pads every row (pad_id trails the real content in
  `input_ids`, with `attention_mask` zeroed over the same trailing span) — pinned down as a
  regression test in the implementation plan.
- Tokenizer wrap: `PreTrainedTokenizerFast(tokenizer_object=<our Tokenizer>, pad_token="[PAD]",
unk_token="[UNK]", eos_token="[PAD]")` — trivial, since `tokenizers.Tokenizer` is literally what
  backs HF fast tokenizers already. `eos_token="[PAD]"` matters beyond labeling: see Dataset
  formatting below.
- Checkpoint loading reuses `checkpoint.py::load_checkpoint(path, wrapper.model)` unchanged — it
  already takes a generic `nn.Module`, so passing the wrapper's inner submodule requires no changes
  to that file.

**Dataset formatting**: verified against current TRL source (`trainer/dpo_trainer.py`'s
`_prepare_dataset`) — `DPOTrainer` _always_ tokenizes from raw `prompt`/`chosen`/`rejected` text;
there is no pre-tokenized `input_ids` bypass (an earlier draft of this spec incorrectly assumed
one). Two dataset formats are supported: "conversational" (`{role, content}` turns, templated via
`tokenizer.apply_chat_template`) and "standard" (plain pre-formatted text). This pipeline uses
**standard**: our tokenizer has no chat template configured, and authoring a Jinja template to
match our exact `<|user|>\n...\n<|assistant|>\n` convention would be unplanned complexity for no
benefit here. `pairs_dpo.jsonl`'s `prompt` field is formatted once, at `training/dpo.py` load
time, as `format_turn("user", question) + "<|assistant|>\n"` (reusing `data/chat.py`'s
`format_turn()` for the user turn); `chosen`/`rejected` stay the raw completion text, unmodified.

For the stop signal: TRL's standard-format path automatically appends `tokenizer.eos_token` to
`chosen`/`rejected` if not already present, before tokenizing (`_prepare_dataset`'s `add_eos`
step) — which is exactly the role `[PAD]` already plays in this repo's SFT convention (a stop
signal spliced after each assistant turn, itself supervised). Setting `eos_token="[PAD]"` on the
wrapped tokenizer means this happens automatically; `training/dpo.py` doesn't need to hand-roll
appending it. **Confirmed empirically** (`train_tokenizer` on a small fake corpus,
`tokenizer.encode("hello world" + PAD_TOKEN).ids`): the literal text `"[PAD]"` round-trips
cleanly to the dedicated pad token id as the final token, with the preceding text's encoding
unaffected — the byte-level BPE tokenizer treats `[PAD]` as an added/special token regardless of
what text immediately precedes it. No fallback is needed; this is pinned down as a regression
test in the implementation plan rather than left as an open question.

**Reference model**: an earlier draft of this spec assumed `DPOTrainer`'s default (`ref_model=None`)
would deep-copy the in-memory policy model. **That's wrong, confirmed by reading current TRL
source** (`trainer/dpo_trainer.py`): when `ref_model is None` and the model isn't a PEFT model,
`DPOTrainer` calls `create_model_from_path(get_config_model_id(self.model.config), ...)` — it
*reloads* a fresh model from a Hub repo id/path derived from the policy model's config, rather than
copying the already-loaded weights. Our wrapped model is constructed directly in Python with no
Hub repo id, so `get_config_model_id` resolves to an empty string and this path raises
`HFValidationError` (confirmed by triggering it against a tiny instance of `TransformerLMForCausalLM`).
The fix, also confirmed working: construct and pass `ref_model=` explicitly — a second
`TransformerLMForCausalLM` instance loaded from the same starting checkpoint weights
(`ref_model.load_state_dict(model.state_dict())` right after both are constructed, before any
training steps run). Passing `ref_model` explicitly takes a different branch in `DPOTrainer.__init__`
that uses it as-is, never touching `create_model_from_path`. The model is small enough (431M
params) that the extra frozen copy's memory cost on the A100 is a non-issue — the "simplest
option" framing from the earlier draft still holds, it's just this explicit-construction form of
it, not `ref_model=None`.

**Hyperparameters** (`DPOConfig`, field names verified against current TRL docs): `beta=0.1`,
`loss_type="sigmoid"` (the original Rafailov et al. DPO loss), `learning_rate` well below SFT's
`3e-5` (DPO norms are typically `~5e-7`-`1e-6` for full fine-tunes), `num_train_epochs=1` (a
~2,000-pair dataset is easy to overfit past one pass), `max_length` matching
`DataConfig.max_seq_len`. Every flag exposed as a CLI arg with its default read from `DPOConfig`'s
own field default, the same single-source-of-truth pattern `train.py` already uses for
`ModelConfig`/`TrainConfig`.

**Closing the loop back to `generate.py`**: after `trainer.train()` finishes, `main()` takes the
trained wrapper's inner `self.model.state_dict()` and calls the existing `save_checkpoint()`
(same `step_N.pt` + persisted `model_config` format pretrain/SFT already produce).
`generate.py`/`s3.py`/checkpoint pruning stay completely unaware DPO ever happened — no new
loading path added anywhere else in the codebase. HF `Trainer`'s own mid-run checkpointing is
disabled (`save_strategy="no"`) given the run's short duration (~2,000 pairs, 1 epoch); a spot
preemption means restarting this short run from scratch rather than building resume support sized
for it.

## Error handling

- **Judge API calls**: bounded retry-with-backoff on transient errors (mirrors
  `checkpoint.py::save_checkpoint`'s existing pattern), then tagged `api_failure` and skipped.
- **RunPod spot preemption**: no resume machinery for any of these three stages, given each runs in
  minutes over a few thousand rows — a preempted run is simply re-launched. A deliberate
  simplicity trade-off given the pipeline's short duration, not an oversight.

## Testing

Same principle as the rest of the repo: everything except the training loop itself gets a fast,
CPU-only unit test with tiny fake data and no network calls.

| File                  | Tested with tiny fake data / mocks                                                                                                                                                                                            | Not automated (manual smoke test)                                                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model/hf_wrapper.py` | forward() shape + `CausalLMOutputWithPast` contract on a tiny `ModelConfig`; tokenizer-wrap encode/decode round-trip incl. `[PAD]` handling; checkpoint load-in via existing `load_checkpoint`                                | —                                                                                                                                                                      |
| `judge.py`            | prompt-template construction; JSON-schema response parsing (valid / malformed / missing-verdict); position-bias agreement logic (agree->keep, disagree->discard+tag); retry-then-`api_failure` logic; length-ratio diagnostic | the live judge API call itself (needs a real call to verify structured-output compliance — that's the startup self-check, exercised manually as part of the pilot run) |
| `generate_pairs.py`   | pipeline orchestration against a fake tiny model + fake prompts, asserting output schema                                                                                                                                      | actual GPU sampling quality (that's what the pilot run is for)                                                                                                         |
| `training/dpo.py`     | CLI arg parsing -> `DPOConfig` field mapping (same pattern as `train.py`'s arg-default consistency)                                                                                                                           | `trainer.train()` orchestration itself, validated via a manual pilot smoke test on a small batch (~10-20 pairs), the same way `train()`/`main()` already are           |

The judge's `InferenceClient` call sits behind a small injectable parameter (not a full plugin
abstraction — just enough that tests pass a fake callable instead of hitting the network),
consistent with no automated test hitting the network anywhere in this pipeline except the
deliberate, manual, real-call startup self-check.

## Open questions carried into the implementation plan

- ~~Exact round-trip behavior of `"[PAD]"` literal text through the tokenizer as an appended
  `eos_token`~~ — resolved (see Dataset formatting above); pinned down as a regression test in
  the implementation plan.
- ~~Confirm TRL's default DPO data collator right-pads~~ — resolved (see the `hf_wrapper.py`
  bullet above); pinned down as a regression test in the implementation plan. Investigating this
  also surfaced and fixed a real bug in this spec's original reference-model design — see
  Reference model above.
- Fallback judge provider/model pair(s) to document for the startup self-check, in case
  `together`/`Llama-3.3-70B-Instruct` stops honoring `response_format` at run time.
