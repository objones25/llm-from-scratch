# DPO Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-stage DPO/RLAIF pipeline (`generate_pairs.py` → `judge.py` → `training/dpo.py`) that preference-tunes the existing SFT checkpoint, plus the `model/hf_wrapper.py` adapter that lets TRL's `DPOTrainer` drive this repo's hand-rolled `TransformerLM` unmodified — per `docs/superpowers/specs/2026-08-18-dpo-pipeline-design.md`.

**Architecture:** `model/hf_wrapper.py` wraps `TransformerLM`/`tokenizers.Tokenizer` as a minimal `transformers.PreTrainedModel`/`PreTrainedTokenizerFast` pair so `DPOTrainer` can be used exactly as documented, with no reimplementation of the DPO loss. `generate_pairs.py` samples two completions per prompt from the SFT checkpoint via `generate.py`'s existing KV-cache generation code. `judge.py` turns those into `(chosen, rejected)` pairs via an external judge model, double-evaluated per pair to catch position bias, with a JSON-schema-constrained response so the verdict is never regex-parsed. `training/dpo.py` trains on the filtered pairs and exports the result back through the existing `save_checkpoint()` format, so `generate.py`/`s3.py`/checkpoint pruning stay completely unaware DPO ever happened.

**Tech Stack:** PyTorch, Hugging Face `transformers`/`trl`/`huggingface_hub` (new), existing `llmtrain` package conventions (dataclass configs, CPU-only unit tests with tiny fake data, `uv`-managed dependencies).

**Spec:** `docs/superpowers/specs/2026-08-18-dpo-pipeline-design.md`

## Global Constraints

- Flat, single-purpose files under `src/llmtrain/` — no nested feature folders (matches `generate.py`/`s3.py`).
- Every CLI flag's default is sourced from the relevant config dataclass's own field default (the pattern `train.py` already uses for `ModelConfig`/`TrainConfig`), except `trl.DPOConfig.loss_type`, which uses `default_factory` and has no class-level default — hardcode `"sigmoid"` as the CLI default instead (confirmed via `dataclasses.fields(DPOConfig)`: the factory returns `["sigmoid"]`).
- Reuse existing code rather than duplicating it: `generate.py`'s `generate_token_ids` (KV-cache generation), `checkpoint.py`'s `save_checkpoint`/`load_checkpoint`, `data/chat.py`'s `format_turn`.
- Do not modify `generate.py`, `checkpoint.py`, or `s3.py` — the whole point of the checkpoint-export step in Task 7 is that those files never need to know DPO happened.
- No automated test may hit the network or a real GPU. The one deliberate exception (the judge's startup self-check) is exercised manually, not by `pytest`.
- `trainer.state.global_step` (not a manually-tracked counter) is the step number used when exporting the final checkpoint — confirmed available after `trainer.train()` via a real `DPOTrainer` run in this plan's verification.
- `DPOConfig(gradient_checkpointing=False, ...)` is **required**, not optional — confirmed by triggering the failure directly: `DPOConfig`'s default (`gradient_checkpointing=True`) makes `Trainer.train()` call `model.gradient_checkpointing_enable()`, which raises `ValueError: TransformerLMForCausalLM does not support gradient checkpointing` because the wrapper never declares that support (matches this repo's own "deferred, not implemented" status for gradient checkpointing).
- `ref_model` must always be passed **explicitly** to `DPOTrainer` — confirmed by triggering the failure directly: leaving it `None` makes current TRL (1.10.0) call `create_model_from_path()` against a Hub repo id derived from the policy model's config, which is empty for our in-memory-constructed model and raises `HFValidationError`. The fix is a second `TransformerLMForCausalLM` instance loaded from the same starting weights.
- `ruff check .`, `uv run mypy src/`, and `uv run pytest` must stay clean after every task.
- Commit after each task (not each step) unless a step says otherwise.

---

## Task 1: Add `transformers` and `trl` dependencies

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (both via `uv add`, not hand-edited)

**Interfaces:**
- Produces: `transformers` and `trl` importable in the project's normal `uv run` environment (not behind an optional extra — Task 3's `hf_wrapper.py` and Task 7's `training/dpo.py` need them for their normal, always-run unit tests, unlike the `cuda`/`s3` extras which gate genuinely optional, environment-specific code).

- [ ] **Step 1: Add the dependencies**

Run: `uv add transformers trl`

This resolves and pins compatible versions of `transformers`, `trl`, and their transitive dependencies (including possibly moving `tokenizers` within the range this project's `pyproject.toml` already allows, `tokenizers>=0.20` — confirmed compatible with the project's own `data/tokenizer.py`/`data/chat.py` usage, which only relies on stable, long-standing `tokenizers` APIs).

- [ ] **Step 2: Verify the imports work**

Run: `uv run python -c "import transformers, trl; print(transformers.__version__, trl.__version__)"`
Expected: prints two version strings, no `ModuleNotFoundError`.

- [ ] **Step 3: Confirm the existing suite is still green**

Run: `uv run pytest -q`
Expected: same pass count as before this task (114 passed, 1 xfailed, plus 2 pre-existing `test_s3.py` failures if `boto3`'s optional `s3` extra isn't installed locally — unrelated to this change).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add transformers and trl dependencies for the DPO pipeline"
```

---

## Task 2: Pin down the `[PAD]`-as-`eos_token` round-trip

**Files:**
- Modify: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: `llmtrain.data.tokenizer.train_tokenizer`, `PAD_TOKEN` (existing).
- Produces: nothing new — this is a regression test pinning down behavior `model/hf_wrapper.py` (Task 3) and `training/dpo.py` (Task 7) both depend on: TRL's DPO pipeline appends `tokenizer.eos_token` (literal text) to `chosen`/`rejected` before encoding, and this repo sets `eos_token="[PAD]"` so that append reinforces the same stop signal SFT already taught. This test proves that mechanism produces the correct single pad token id, not garbled/split tokens.

- [ ] **Step 1: Write the test**

Add to `tests/test_tokenizer.py`:

```python
from llmtrain.data.tokenizer import PAD_TOKEN, encode_batch, train_tokenizer


def test_pad_token_literal_text_round_trips_to_pad_id_when_appended():
    # TRL's DPOTrainer appends `tokenizer.eos_token` (literal text, not a raw id) to
    # chosen/rejected completions before encoding (see docs/superpowers/specs/
    # 2026-08-18-dpo-pipeline-design.md's "Dataset formatting" section). Setting
    # eos_token="[PAD]" on the wrapped tokenizer (model/hf_wrapper.py) only reinforces
    # the SFT-taught stop signal if this literal text round-trips to the single
    # dedicated pad token id, matching how encode_chat_example appends it as a raw id.
    tokenizer = train_tokenizer(["hello world", "hello there", "the quick brown fox"], vocab_size=50)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    plain_ids = tokenizer.encode("hello world").ids
    ids_with_pad_appended = tokenizer.encode("hello world" + PAD_TOKEN).ids

    assert ids_with_pad_appended[-1] == pad_id
    assert ids_with_pad_appended[:-1] == plain_ids
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_tokenizer.py::test_pad_token_literal_text_round_trips_to_pad_id_when_appended -v`
Expected: PASS (this behavior was confirmed manually before writing this plan; the test pins it down as a permanent regression check).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tokenizer.py
git commit -m "Pin down PAD-token-as-eos_token round-trip for DPO's dataset formatting"
```

---

## Task 3: `model/hf_wrapper.py` — HF-compatible adapter around `TransformerLM`

**Files:**
- Create: `src/llmtrain/model/hf_wrapper.py`
- Test: Create `tests/test_hf_wrapper.py`

**Interfaces:**
- Consumes: `llmtrain.model.transformer.TransformerLM`, `llmtrain.training.config.ModelConfig`, `llmtrain.data.tokenizer.PAD_TOKEN`/`UNK_TOKEN` (existing).
- Produces: `TransformerLMConfig(PretrainedConfig)` with `to_model_config() -> ModelConfig` and `from_model_config(ModelConfig) -> TransformerLMConfig` (classmethod); `TransformerLMForCausalLM(PreTrainedModel)` holding `self.model: TransformerLM`, `forward(input_ids, attention_mask=None, **kwargs) -> CausalLMOutputWithPast`; `wrap_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast`. Task 4's verification test, and Task 7 (`training/dpo.py`), both import all three names from this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hf_wrapper.py`:

```python
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmtrain.data.tokenizer import PAD_TOKEN, train_tokenizer
from llmtrain.model.hf_wrapper import TransformerLMConfig, TransformerLMForCausalLM, wrap_tokenizer
from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint
from llmtrain.training.config import ModelConfig


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)


def test_forward_returns_causal_lm_output_with_correct_logits_shape():
    hf_config = TransformerLMConfig.from_model_config(_tiny_model_config())
    model = TransformerLMForCausalLM(hf_config)
    input_ids = torch.randint(0, 16, (3, 6))

    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert isinstance(output, CausalLMOutputWithPast)
    assert output.logits.shape == (3, 6, 16)


def test_forward_ignores_attention_mask_and_still_produces_gradients():
    hf_config = TransformerLMConfig.from_model_config(_tiny_model_config())
    model = TransformerLMForCausalLM(hf_config)
    input_ids = torch.randint(0, 16, (2, 5))

    output = model(input_ids=input_ids, attention_mask=None)
    output.logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_wrap_tokenizer_preserves_encode_decode_round_trip_and_sets_special_tokens():
    tokenizer = train_tokenizer(["hello world", "hello there"], vocab_size=50)
    wrapped = wrap_tokenizer(tokenizer)

    assert wrapped("hello world")["input_ids"] == tokenizer.encode("hello world").ids
    assert wrapped.decode(tokenizer.encode("hello world").ids) == "hello world"
    assert wrapped.pad_token == PAD_TOKEN
    assert wrapped.eos_token == PAD_TOKEN
    assert wrapped.pad_token_id == tokenizer.token_to_id(PAD_TOKEN)


def test_wrapper_round_trips_through_the_existing_checkpoint_format(tmp_path):
    model_cfg = _tiny_model_config()
    hf_config = TransformerLMConfig.from_model_config(model_cfg)
    original = TransformerLMForCausalLM(hf_config)
    optimizer = torch.optim.AdamW(original.model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, original.model, optimizer, step=1)

    loaded = TransformerLMForCausalLM(hf_config)
    load_checkpoint(checkpoint_path, loaded.model)

    for p_orig, p_loaded in zip(original.model.parameters(), loaded.model.parameters()):
        assert torch.equal(p_orig, p_loaded)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_hf_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.model.hf_wrapper'`.

- [ ] **Step 3: Write `model/hf_wrapper.py`**

Create `src/llmtrain/model/hf_wrapper.py`:

```python
from tokenizers import Tokenizer
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmtrain.data.tokenizer import PAD_TOKEN, UNK_TOKEN
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import ModelConfig


class TransformerLMConfig(PretrainedConfig):
    model_type = "transformer_lm"

    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 1440,
        n_layers: int = 20,
        n_heads: int = 20,
        n_kv_heads: int = 4,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.rope_theta = rope_theta
        # Weight tying is already handled directly on the parameter in
        # TransformerLM.__init__ (self.head.weight = self.token_emb.weight) -- HF's own
        # tie-weights machinery should stay inert rather than fight it.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            dropout=self.dropout,
            rope_theta=self.rope_theta,
        )

    @classmethod
    def from_model_config(cls, model_config: ModelConfig) -> "TransformerLMConfig":
        return cls(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
            n_kv_heads=model_config.n_kv_heads,
            dropout=model_config.dropout,
            rope_theta=model_config.rope_theta,
        )


class TransformerLMForCausalLM(PreTrainedModel):
    config_class = TransformerLMConfig

    def __init__(self, config: TransformerLMConfig) -> None:
        super().__init__(config)
        self.model = TransformerLM(config.to_model_config())

    def forward(
        self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor | None" = None, **kwargs
    ) -> CausalLMOutputWithPast:
        # attention_mask is accepted (required by HF's calling convention, and DPOTrainer
        # always passes one) but deliberately not forwarded: TransformerLM.forward() has no
        # such parameter, relying purely on causal ordering. This is only correct because
        # DPO's batches are right-padded (confirmed in Task 4) -- causal masking alone then
        # prevents any real token from attending into the padded tail, the same property
        # make_collate_fn's SFT collation already relies on.
        logits = self.model(input_ids)
        return CausalLMOutputWithPast(logits=logits)


def wrap_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    # eos_token="[PAD]" matters beyond labeling: TRL's DPOTrainer appends
    # tokenizer.eos_token (literal text) to chosen/rejected completions before encoding,
    # which is exactly the stop-signal role [PAD] already plays in this repo's SFT
    # convention (see tests/test_tokenizer.py's PAD-round-trip regression test).
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        eos_token=PAD_TOKEN,
    )
```

Add `import torch` at the top of the file (needed for the `forward` type hints above; written as string literals in this plan only to keep the diff readable — use a real top-of-file `import torch` in the actual file, not string-quoted annotations).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_hf_wrapper.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/model/hf_wrapper.py tests/test_hf_wrapper.py
git commit -m "Add HF-compatible wrapper around TransformerLM for TRL's DPOTrainer"
```

---

## Task 4: Verify `hf_wrapper.py` against a real `DPOTrainer` (right-padding + `ref_model` requirement)

**Files:**
- Modify: `tests/test_hf_wrapper.py`

**Interfaces:**
- Consumes: `TransformerLMConfig`, `TransformerLMForCausalLM`, `wrap_tokenizer` (Task 3); `trl.DPOConfig`/`DPOTrainer`, `datasets.Dataset` (new).
- Produces: nothing new — this locks in, as permanent regression tests, the two facts this plan's own verification already surfaced: (1) DPO batches are right-padded, which is what makes `hf_wrapper.py`'s attention-mask-ignoring `forward()` correct; (2) omitting `ref_model` fails for this wrapper, which is why Task 7 always constructs one explicitly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hf_wrapper.py`:

```python
import pytest
from datasets import Dataset
from trl import DPOConfig, DPOTrainer


def _tiny_dpo_setup():
    texts = [
        "hello world",
        "hello there friend",
        "the quick brown fox jumps over",
        "goodbye my old friend",
    ]
    tokenizer = train_tokenizer(texts, vocab_size=64)
    wrapped_tokenizer = wrap_tokenizer(tokenizer)
    hf_config = TransformerLMConfig(
        vocab_size=wrapped_tokenizer.vocab_size, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1
    )
    model = TransformerLMForCausalLM(hf_config)
    ref_model = TransformerLMForCausalLM(hf_config)
    ref_model.load_state_dict(model.state_dict())
    dataset = Dataset.from_dict(
        {
            "prompt": ["hello ", "hello ", "hello ", "hello "],
            "chosen": ["world", "there friend, the quick brown fox jumps over", "world", "there friend"],
            "rejected": ["there", "world", "there", "world"],
        }
    )
    return model, ref_model, wrapped_tokenizer, dataset


def test_dpo_trainer_batches_are_right_padded(tmp_path):
    model, ref_model, wrapped_tokenizer, dataset = _tiny_dpo_setup()
    dpo_config = DPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        report_to=[],
        use_cpu=True,
        max_length=64,
        gradient_checkpointing=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=wrapped_tokenizer,
    )

    batch = next(iter(trainer.get_train_dataloader()))

    # Right-padding: once a row's attention_mask hits 0 (padded), every later position in
    # that row must also be 0 -- the mask never goes back to 1 after the first pad.
    for row in batch["attention_mask"]:
        seen_pad = False
        for value in row.tolist():
            if value == 0:
                seen_pad = True
            elif seen_pad:
                pytest.fail("attention_mask has a real token after a padded position")


def test_dpo_trainer_requires_an_explicit_ref_model_for_this_wrapper(tmp_path):
    # Regression test for a real, confirmed bug: DPOTrainer's ref_model=None default tries
    # to reload a fresh reference model from a Hub repo id derived from the policy model's
    # config (create_model_from_path). Our wrapper is constructed directly in Python with
    # no Hub repo id, so that path raises -- which is exactly why training/dpo.py always
    # builds and passes a second TransformerLMForCausalLM instance explicitly.
    model, _ref_model, wrapped_tokenizer, dataset = _tiny_dpo_setup()
    dpo_config = DPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        report_to=[],
        use_cpu=True,
        max_length=64,
        gradient_checkpointing=False,
    )

    with pytest.raises(Exception):
        DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=wrapped_tokenizer,
        )
```

- [ ] **Step 2: Run the tests to verify they fail for the right reason**

Run: `uv run pytest tests/test_hf_wrapper.py::test_dpo_trainer_batches_are_right_padded tests/test_hf_wrapper.py::test_dpo_trainer_requires_an_explicit_ref_model_for_this_wrapper -v`
Expected: both already pass, since Task 3's `hf_wrapper.py` and Task 1's dependencies are both already in place — this task adds no new production code, only pins down behavior already verified manually while writing this plan. If either fails, that means the installed `trl`/`transformers` versions behave differently than the versions verified here (`transformers==5.15.0`, `trl==1.10.0`) — re-diagnose against the actual installed versions rather than assuming this plan's text is still accurate.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hf_wrapper.py
git commit -m "Pin down DPO batch right-padding and explicit-ref_model requirement"
```

---

## Task 5: `judge.py` — LLM-as-judge with position-bias double-evaluation

**Files:**
- Create: `src/llmtrain/judge.py`
- Test: Create `tests/test_judge.py`

**Interfaces:**
- Consumes: `huggingface_hub.InferenceClient` (new, already a transitive dependency of `datasets`/`tokenizers`, confirmed importable without any new install).
- Produces: `JUDGE_JSON_SCHEMA: dict`, `JudgeParseError(Exception)`, `build_judge_messages(prompt, response_a, response_b) -> list[dict]`, `call_judge(client, model, prompt, response_a, response_b, temperature) -> dict`, `call_judge_with_retry(client, model, prompt, response_a, response_b, temperature, max_attempts, retry_delay) -> dict | None`, `JudgeResult` (dataclass: `kept: bool`, `chosen: str | None`, `rejected: str | None`, `discard_reason: str | None`, `length_ratio: float | None`), `judge_pair(client, model, prompt, completion_a, completion_b, temperature) -> JudgeResult`, `run_judge_pipeline(client, model, rows, temperature) -> tuple[list[dict], dict]`. Task 7 does not import from this module (it consumes `judge.py`'s *output file*, `pairs_dpo.jsonl`, not its Python API).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge.py`:

```python
import json

import pytest

from llmtrain.judge import (
    JudgeParseError,
    call_judge,
    call_judge_with_retry,
    judge_pair,
    run_judge_pipeline,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses: list) -> None:
        self._responses = iter(responses)

    def create(self, **kwargs):
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return _FakeCompletion(response)


class _FakeChat:
    def __init__(self, responses: list) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.chat = _FakeChat(responses)


def _verdict(verdict: str, reasoning: str = "because") -> str:
    return json.dumps({"reasoning": reasoning, "verdict": verdict})


def test_call_judge_parses_a_valid_response():
    client = _FakeClient([_verdict("A")])
    result = call_judge(client, "some-model", "prompt", "resp a", "resp b")
    assert result["verdict"] == "A"


def test_call_judge_raises_judge_parse_error_on_malformed_json():
    client = _FakeClient(["not json"])
    with pytest.raises(JudgeParseError):
        call_judge(client, "some-model", "prompt", "resp a", "resp b")


def test_call_judge_raises_judge_parse_error_on_missing_verdict():
    client = _FakeClient([json.dumps({"reasoning": "because"})])
    with pytest.raises(JudgeParseError):
        call_judge(client, "some-model", "prompt", "resp a", "resp b")


def test_call_judge_with_retry_recovers_from_a_transient_failure():
    client = _FakeClient([RuntimeError("timeout"), _verdict("B")])
    result = call_judge_with_retry(
        client, "some-model", "prompt", "resp a", "resp b", max_attempts=3, retry_delay=0.0
    )
    assert result["verdict"] == "B"


def test_call_judge_with_retry_gives_up_after_max_attempts():
    client = _FakeClient([RuntimeError("timeout")] * 3)
    result = call_judge_with_retry(
        client, "some-model", "prompt", "resp a", "resp b", max_attempts=3, retry_delay=0.0
    )
    assert result is None


def test_judge_pair_keeps_the_pair_when_both_orderings_agree():
    # forward call: A=completion_a, B=completion_b -> "A" means completion_a won
    # swapped call: A=completion_b, B=completion_a -> "B" means completion_a won again
    client = _FakeClient([_verdict("A"), _verdict("B")])
    result = judge_pair(client, "some-model", "prompt", "completion_a text", "completion_b text")
    assert result.kept
    assert result.chosen == "completion_a text"
    assert result.rejected == "completion_b text"
    assert result.length_ratio is not None


def test_judge_pair_discards_on_position_bias_disagreement():
    # forward call: A=completion_a, B=completion_b -> "A" means completion_a won
    # swapped call: A=completion_b, B=completion_a -> "A" means completion_b won this time
    client = _FakeClient([_verdict("A"), _verdict("A")])
    result = judge_pair(client, "some-model", "prompt", "completion_a text", "completion_b text")
    assert not result.kept
    assert result.discard_reason == "position_bias_disagreement"


def test_judge_pair_discards_on_parse_failure():
    client = _FakeClient(["not json", _verdict("A")])
    result = judge_pair(client, "some-model", "prompt", "completion_a text", "completion_b text")
    assert not result.kept
    assert result.discard_reason == "parse_failure"


def test_judge_pair_discards_on_api_failure():
    client = _FakeClient([RuntimeError("timeout")] * 6)
    result = judge_pair(
        client,
        "some-model",
        "prompt",
        "completion_a text",
        "completion_b text",
    )
    assert not result.kept
    assert result.discard_reason == "api_failure"


def test_run_judge_pipeline_tallies_kept_and_discard_counts():
    # row 1: agree -> kept. row 2: disagree -> position_bias_disagreement.
    client = _FakeClient([_verdict("A"), _verdict("B"), _verdict("A"), _verdict("A")])
    rows = [
        {"prompt": "p1", "completion_a": "a1", "completion_b": "b1"},
        {"prompt": "p2", "completion_a": "a2", "completion_b": "b2"},
    ]
    kept, summary = run_judge_pipeline(client, "some-model", rows)
    assert summary["total"] == 2
    assert summary["kept"] == 1
    assert summary["discard_counts"]["position_bias_disagreement"] == 1
    assert kept == [{"prompt": "p1", "chosen": "a1", "rejected": "b1"}]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.judge'`.

- [ ] **Step 3: Write `judge.py`**

Create `src/llmtrain/judge.py`:

```python
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import InferenceClient

from llmtrain.logging_config import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_PROVIDER = "together"
DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_JUDGE_TEMPERATURE = 0.15
_MAX_JUDGE_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 5.0

JUDGE_JSON_SCHEMA = {
    "name": "judge_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "verdict": {"type": "string", "enum": ["A", "B"]},
        },
        "required": ["reasoning", "verdict"],
        "additionalProperties": False,
    },
    "strict": True,
}

JUDGE_PROMPT_TEMPLATE = """You are evaluating two AI assistant responses to the same user prompt.

Guidelines:
- Judge only correctness, helpfulness, and relevance to the prompt.
- Do NOT prefer a response merely because it is longer or more detailed. A shorter, correct,
  on-point answer beats a longer one that pads with unnecessary detail or repetition. For
  example, given "What is 2+2?", the response "4" is better than a paragraph explaining basic
  arithmetic before arriving at "4".
- You must pick exactly one response as better. Ties are not allowed -- if genuinely close,
  break the tie on correctness, then relevance, then clarity, in that order.
- First write 1-2 sentences of reasoning, then give your verdict.

Prompt: {prompt}
Response A: {response_a}
Response B: {response_b}"""


class JudgeParseError(Exception):
    pass


def build_judge_messages(prompt: str, response_a: str, response_b: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": JUDGE_PROMPT_TEMPLATE.format(
                prompt=prompt, response_a=response_a, response_b=response_b
            ),
        }
    ]


def call_judge(
    client: InferenceClient,
    model: str,
    prompt: str,
    response_a: str,
    response_b: str,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
) -> dict:
    completion = client.chat.completions.create(
        model=model,
        messages=build_judge_messages(prompt, response_a, response_b),
        response_format={"type": "json_schema", "json_schema": JUDGE_JSON_SCHEMA},
        temperature=temperature,
    )
    content = completion.choices[0].message.content
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JudgeParseError(f"judge response was not valid JSON: {content!r}") from exc
    if parsed.get("verdict") not in ("A", "B"):
        raise JudgeParseError(f"judge response missing/invalid 'verdict': {parsed!r}")
    return parsed


def call_judge_with_retry(
    client: InferenceClient,
    model: str,
    prompt: str,
    response_a: str,
    response_b: str,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    max_attempts: int = _MAX_JUDGE_ATTEMPTS,
    retry_delay: float = _RETRY_DELAY_SECONDS,
) -> dict | None:
    for attempt in range(1, max_attempts + 1):
        try:
            return call_judge(client, model, prompt, response_a, response_b, temperature)
        except JudgeParseError:
            # Not transient -- a malformed/off-schema response won't fix itself on retry.
            raise
        except Exception:
            if attempt == max_attempts:
                return None
            logger.warning(
                "judge call failed on attempt %d/%d, retrying",
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(retry_delay)
    return None


@dataclass
class JudgeResult:
    kept: bool
    chosen: str | None = None
    rejected: str | None = None
    discard_reason: str | None = None
    length_ratio: float | None = None


def judge_pair(
    client: InferenceClient,
    model: str,
    prompt: str,
    completion_a: str,
    completion_b: str,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
) -> JudgeResult:
    try:
        forward = call_judge_with_retry(client, model, prompt, completion_a, completion_b, temperature)
        swapped = call_judge_with_retry(client, model, prompt, completion_b, completion_a, temperature)
    except JudgeParseError:
        return JudgeResult(kept=False, discard_reason="parse_failure")

    if forward is None or swapped is None:
        return JudgeResult(kept=False, discard_reason="api_failure")

    # forward: A=completion_a, B=completion_b -> map the verdict back to which original
    # completion won. swapped: A=completion_b, B=completion_a -> same mapping, reversed.
    forward_winner = completion_a if forward["verdict"] == "A" else completion_b
    swapped_winner = completion_b if swapped["verdict"] == "A" else completion_a

    if forward_winner != swapped_winner:
        return JudgeResult(kept=False, discard_reason="position_bias_disagreement")

    winner = forward_winner
    loser = completion_b if winner == completion_a else completion_a
    # Character length, not token length -- a cheap diagnostic, not precise, and judge.py
    # has no reason to depend on this repo's tokenizer.
    length_ratio = len(winner) / max(len(loser), 1)
    return JudgeResult(kept=True, chosen=winner, rejected=loser, length_ratio=length_ratio)


def run_judge_pipeline(
    client: InferenceClient,
    model: str,
    rows: list[dict],
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    discard_counts = {"position_bias_disagreement": 0, "parse_failure": 0, "api_failure": 0}
    length_ratios: list[float] = []
    for row in rows:
        result = judge_pair(
            client, model, row["prompt"], row["completion_a"], row["completion_b"], temperature
        )
        if result.kept:
            kept.append({"prompt": row["prompt"], "chosen": result.chosen, "rejected": result.rejected})
            assert result.length_ratio is not None
            length_ratios.append(result.length_ratio)
        else:
            assert result.discard_reason is not None
            discard_counts[result.discard_reason] += 1
    summary = {
        "total": len(rows),
        "kept": len(kept),
        "discard_counts": discard_counts,
        "mean_length_ratio": sum(length_ratios) / len(length_ratios) if length_ratios else None,
    }
    return kept, summary


def _startup_self_check(client: InferenceClient, model: str, temperature: float) -> None:
    result = call_judge_with_retry(
        client,
        model,
        "What is 2+2?",
        "4",
        "The answer is definitely 5, I am very confident.",
        temperature,
    )
    if result is None:
        raise RuntimeError(
            f"judge startup self-check failed: no valid structured-output response from "
            f"provider/model {model!r}. Try a different --judge-provider/--judge-model pair."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge sampled completion pairs into DPO preference pairs"
    )
    parser.add_argument("--input", type=str, required=True, help="path to pairs_raw.jsonl")
    parser.add_argument("--output", type=str, required=True, help="path to write pairs_dpo.jsonl")
    parser.add_argument("--judge-provider", type=str, default=DEFAULT_JUDGE_PROVIDER)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_JUDGE_TEMPERATURE)
    parser.add_argument("--log-file", type=str, default="app.log")
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)
    client = InferenceClient(provider=args.judge_provider, api_key=os.environ["HF_TOKEN"])
    _startup_self_check(client, args.judge_model, args.temperature)

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    kept, summary = run_judge_pipeline(client, args.judge_model, rows, args.temperature)

    with open(args.output, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    logger.info("judge pipeline complete", extra=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_judge.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/judge.py tests/test_judge.py
git commit -m "Add LLM-as-judge stage with position-bias double-evaluation"
```

---

## Task 6: `generate_pairs.py` — sample completion pairs from the SFT checkpoint

**Files:**
- Create: `src/llmtrain/generate_pairs.py`
- Test: Create `tests/test_generate_pairs.py`

**Interfaces:**
- Consumes: `llmtrain.generate.generate_token_ids` (existing, reused not duplicated), `llmtrain.data.chat.format_turn` (existing), `llmtrain.training.config.GenerationConfig`/`ModelConfig` (existing).
- Produces: `format_prompt(question: str) -> str`, `sample_completion(model, tokenizer, question, config) -> str`, `generate_pairs(model, tokenizer, questions: list[str], config) -> list[dict]`. No other task imports from this module (Task 5's `judge.py` consumes its output file, `pairs_raw.jsonl`, not its Python API).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_pairs.py`:

```python
import torch

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.generate_pairs import format_prompt, generate_pairs, sample_completion
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import GenerationConfig, ModelConfig


def _tiny_setup() -> tuple[TransformerLM, "Tokenizer"]:
    tokenizer = train_tokenizer(
        ["<|user|>\nhello\n<|assistant|>\nhi there\n", "<|user|>\nbye\n<|assistant|>\nsee you\n"],
        vocab_size=64,
    )
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0
    )
    model = TransformerLM(config)
    return model, tokenizer


def test_format_prompt_wraps_question_in_user_and_assistant_tags():
    assert format_prompt("hello") == "<|user|>\nhello\n<|assistant|>\n"


def test_sample_completion_excludes_the_prompt_prefix():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.0)
    completion = sample_completion(model, tokenizer, "hello", config)
    assert "<|user|>" not in completion
    assert "<|assistant|>" not in completion


def test_generate_pairs_produces_two_completions_per_question():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.7)
    rows = generate_pairs(model, tokenizer, ["hello", "bye"], config)
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == {"prompt", "completion_a", "completion_b"}
```

Remove the `"Tokenizer"` string-quoted forward reference from the test file's type hint (written that way in this plan only to avoid an extra unused import in the plan's code block) — in the actual file, either drop the annotation or `from tokenizers import Tokenizer` at the top.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generate_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.generate_pairs'`.

- [ ] **Step 3: Write `generate_pairs.py`**

Create `src/llmtrain/generate_pairs.py`:

```python
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from llmtrain.data.chat import format_turn
from llmtrain.generate import generate_token_ids
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import TransformerLM
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import load_checkpoint
from llmtrain.training.config import GenerationConfig, ModelConfig

PROMPT_DATASET = "trl-lib/ultrafeedback-prompt"


def format_prompt(question: str) -> str:
    return format_turn("user", question) + "<|assistant|>\n"


def sample_completion(
    model: TransformerLM, tokenizer: Tokenizer, question: str, config: GenerationConfig
) -> str:
    prompt = format_prompt(question)
    prompt_ids = tokenizer.encode(prompt).ids
    full_ids = generate_token_ids(model, tokenizer, prompt, config)
    return tokenizer.decode(full_ids[len(prompt_ids) :])


def generate_pairs(
    model: TransformerLM,
    tokenizer: Tokenizer,
    questions: list[str],
    config: GenerationConfig,
) -> list[dict]:
    rows = []
    for question in questions:
        completion_a = sample_completion(model, tokenizer, question, config)
        completion_b = sample_completion(model, tokenizer, question, config)
        rows.append(
            {"prompt": question, "completion_a": completion_a, "completion_b": completion_b}
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample completion pairs for DPO judging")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--output", type=str, required=True, help="path to write pairs_raw.jsonl")
    parser.add_argument("--num-prompts", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=GenerationConfig.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=GenerationConfig.temperature)
    parser.add_argument(
        "--repetition-penalty", type=float, default=GenerationConfig.repetition_penalty
    )
    parser.add_argument("--top-k", type=int, default=GenerationConfig.top_k)
    parser.add_argument("--top-p", type=float, default=GenerationConfig.top_p)
    parser.add_argument("--log-file", type=str, default="app.log")
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)

    checkpoint_path = resolve_local_path(args.checkpoint)
    tokenizer_uri = args.tokenizer_path or sibling_path(args.checkpoint, "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(resolve_local_path(tokenizer_uri)))

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_model_config = raw_checkpoint.get("model_config")
    model_cfg = (
        ModelConfig(**{**saved_model_config, "vocab_size": tokenizer.get_vocab_size()})
        if saved_model_config is not None
        else ModelConfig(vocab_size=tokenizer.get_vocab_size())
    )
    del raw_checkpoint

    model = TransformerLM(model_cfg)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    dataset = load_dataset(PROMPT_DATASET, split="train", streaming=True)
    questions = [row["prompt"][0]["content"] for row in dataset.take(args.num_prompts)]

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    rows = generate_pairs(model, tokenizer, questions, config)

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
```

Note: the checkpoint-loading block (`resolve_local_path`/`sibling_path`/reconstruct `model_cfg` from the checkpoint's persisted config) duplicates a small block already present in `generate.py`'s `main()`. This is deliberate, not an oversight — `generate.py` is off-limits for changes per this plan's Global Constraints, so there's no shared helper to extract without touching it, and the duplicated block is already precedented once (between `train.py`'s `--init-from-checkpoint` path and `generate.py`'s `main()`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_generate_pairs.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llmtrain/generate_pairs.py tests/test_generate_pairs.py
git commit -m "Add generate_pairs.py: sample completion pairs from the SFT checkpoint"
```

---

## Task 7: `training/dpo.py` — DPO training entry point

**Files:**
- Create: `src/llmtrain/training/dpo.py`
- Test: Create `tests/test_dpo_cli.py`

**Interfaces:**
- Consumes: `model/hf_wrapper.py`'s `TransformerLMConfig`/`TransformerLMForCausalLM`/`wrap_tokenizer` (Task 3), `data/chat.py`'s `format_turn` (existing), `checkpoint.py`'s `save_checkpoint`/`load_checkpoint` (existing, unmodified), `s3.py`'s `resolve_local_path`/`sibling_path` (existing, unmodified).
- Produces: `load_dpo_dataset(path) -> datasets.Dataset`, `build_model_and_tokenizer(checkpoint_path, tokenizer_path) -> tuple[TransformerLMForCausalLM, TransformerLMForCausalLM, PreTrainedTokenizerFast, Tokenizer]` (policy model, ref model, wrapped tokenizer, raw tokenizer), `export_checkpoint(model, tokenizer, checkpoint_dir, step) -> None`, `build_dpo_config_from_args(args) -> trl.DPOConfig`. No other task imports from this module — it's the pipeline's terminal stage.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_cli.py`. Per this repo's testing convention (`train.py`'s `train()`/`main()` orchestration has no automated test by design — see `CLAUDE.md`'s Testing strategy and this plan's Task 8), this test covers only CLI arg parsing and dataset formatting, never `trainer.train()` itself:

```python
import json

from llmtrain.training.dpo import build_dpo_config_from_args, load_dpo_dataset


def test_load_dpo_dataset_formats_prompt_with_chat_tags(tmp_path):
    pairs_path = tmp_path / "pairs_dpo.jsonl"
    pairs_path.write_text(
        json.dumps({"prompt": "what is 2+2", "chosen": "4", "rejected": "5"}) + "\n"
    )

    dataset = load_dpo_dataset(pairs_path)

    assert len(dataset) == 1
    row = dataset[0]
    assert row["prompt"] == "<|user|>\nwhat is 2+2\n<|assistant|>\n"
    assert row["chosen"] == "4"
    assert row["rejected"] == "5"


def test_load_dpo_dataset_reads_multiple_lines(tmp_path):
    pairs_path = tmp_path / "pairs_dpo.jsonl"
    rows = [
        {"prompt": "q1", "chosen": "a1", "rejected": "r1"},
        {"prompt": "q2", "chosen": "a2", "rejected": "r2"},
    ]
    pairs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    dataset = load_dpo_dataset(pairs_path)

    assert len(dataset) == 2


def test_build_dpo_config_from_args_maps_cli_flags_to_dpo_config_fields():
    import argparse

    args = argparse.Namespace(
        checkpoint_dir="/tmp/out",
        beta=0.2,
        loss_type="sigmoid",
        learning_rate=1e-6,
        num_train_epochs=2,
        max_length=512,
        batch_size=8,
    )

    dpo_config = build_dpo_config_from_args(args)

    assert dpo_config.output_dir == "/tmp/out"
    assert dpo_config.beta == 0.2
    assert dpo_config.loss_type == ["sigmoid"]
    assert dpo_config.learning_rate == 1e-6
    assert dpo_config.num_train_epochs == 2
    assert dpo_config.max_length == 512
    assert dpo_config.per_device_train_batch_size == 8
    assert dpo_config.gradient_checkpointing is False
    assert dpo_config.save_strategy == "no"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dpo_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmtrain.training.dpo'`.

- [ ] **Step 3: Write `training/dpo.py`**

Create `src/llmtrain/training/dpo.py`:

```python
import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from trl import DPOConfig, DPOTrainer

from llmtrain.data.chat import format_turn
from llmtrain.logging_config import configure_logging
from llmtrain.model.hf_wrapper import TransformerLMConfig, TransformerLMForCausalLM, wrap_tokenizer
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint
from llmtrain.training.config import ModelConfig

logger = logging.getLogger(__name__)


def load_dpo_dataset(path: str | Path) -> Dataset:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    formatted = [
        {
            "prompt": format_turn("user", row["prompt"]) + "<|assistant|>\n",
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in rows
    ]
    return Dataset.from_list(formatted)


def build_model_and_tokenizer(
    checkpoint_path: Path, tokenizer_path: Path
) -> tuple[TransformerLMForCausalLM, TransformerLMForCausalLM, "PreTrainedTokenizerFast", Tokenizer]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_model_config = raw_checkpoint.get("model_config")
    model_cfg = (
        ModelConfig(**{**saved_model_config, "vocab_size": tokenizer.get_vocab_size()})
        if saved_model_config is not None
        else ModelConfig(vocab_size=tokenizer.get_vocab_size())
    )
    del raw_checkpoint

    hf_config = TransformerLMConfig.from_model_config(model_cfg)
    model = TransformerLMForCausalLM(hf_config)
    load_checkpoint(checkpoint_path, model.model)

    # DPOTrainer's ref_model=None default fails for this wrapper (see
    # docs/superpowers/specs/2026-08-18-dpo-pipeline-design.md's "Reference model" section
    # and tests/test_hf_wrapper.py's regression test) -- always build one explicitly.
    ref_model = TransformerLMForCausalLM(hf_config)
    ref_model.load_state_dict(model.state_dict())

    wrapped_tokenizer = wrap_tokenizer(tokenizer)
    return model, ref_model, wrapped_tokenizer, tokenizer


def export_checkpoint(
    model: TransformerLMForCausalLM, tokenizer: Tokenizer, checkpoint_dir: Path, step: int
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # A throwaway optimizer -- save_checkpoint requires one, but generate.py (the only
    # consumer of this checkpoint) never loads optimizer state back (see checkpoint.py's
    # load_checkpoint: optimizer is optional on load, by design, for inference callers).
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
    save_checkpoint(checkpoint_dir / f"step_{step}.pt", model.model, optimizer, step=step)
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    logger.info("exported DPO checkpoint at step %d", step, extra={"step": step})


def build_dpo_config_from_args(args: argparse.Namespace) -> DPOConfig:
    return DPOConfig(
        output_dir=args.checkpoint_dir,
        beta=args.beta,
        loss_type=[args.loss_type],
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        # Our wrapper doesn't implement gradient checkpointing support (matches this
        # project's "deferred, not implemented" status for it elsewhere) -- DPOConfig's own
        # default (True) makes Trainer.train() call model.gradient_checkpointing_enable(),
        # which raises for any model that doesn't declare support for it.
        gradient_checkpointing=False,
        # A single short run (~2,000 pairs, 1 epoch) -- HF Trainer's own mid-run
        # checkpointing is unnecessary; export_checkpoint() saves the final result once,
        # through the existing checkpoint format, after trainer.train() completes.
        save_strategy="no",
        report_to=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO-tune a checkpoint on judged preference pairs")
    parser.add_argument("--checkpoint", type=str, required=True, help="SFT checkpoint to start from")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--pairs", type=str, required=True, help="path to pairs_dpo.jsonl")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="output directory")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-type", type=str, default="sigmoid")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-file", type=str, default="app.log")
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)

    checkpoint_path = resolve_local_path(args.checkpoint)
    tokenizer_uri = args.tokenizer_path or sibling_path(args.checkpoint, "tokenizer.json")
    tokenizer_path = resolve_local_path(tokenizer_uri)

    model, ref_model, wrapped_tokenizer, tokenizer = build_model_and_tokenizer(
        checkpoint_path, tokenizer_path
    )
    dataset = load_dpo_dataset(args.pairs)
    dpo_config = build_dpo_config_from_args(args)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=wrapped_tokenizer,
    )
    trainer.train()

    export_checkpoint(model, tokenizer, Path(args.checkpoint_dir), step=trainer.state.global_step)


if __name__ == "__main__":
    main()
```

Drop the string-quoted `"PreTrainedTokenizerFast"` forward reference in `build_model_and_tokenizer`'s signature in the actual file — either import it from `transformers` at the top or leave the return type as `tuple[...]` without the quotes once the import is present.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dpo_cli.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Run the full suite and lint/type checks**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src/`
Expected: everything clean (same pre-existing `test_s3.py` failures as Task 1 if `boto3` isn't installed locally — unrelated).

- [ ] **Step 6: Commit**

```bash
git add src/llmtrain/training/dpo.py tests/test_dpo_cli.py
git commit -m "Add training/dpo.py: DPO training entry point with checkpoint export"
```

---

## Task 8: Manual pilot smoke test + docs

**Files:**
- Modify: `CLAUDE.md` (Architecture file table + new CLI command block, matching how `generate.py` is documented there)
- Modify: `docs/training-guide.md` (add a "DPO pipeline" walkthrough section, matching the existing Parts structure)

**Interfaces:** None — this task validates the pipeline end-to-end on real (small-scale) infrastructure and updates documentation. No new Python code.

- [ ] **Step 1: Run a small pilot end-to-end, on the RunPod pod (or locally if the SFT checkpoint is available locally)**

```bash
uv run --env-file .env python -m llmtrain.generate_pairs \
    --checkpoint my_sft_checkpoints_v2/step_12000.pt \
    --output pairs_raw.jsonl \
    --num-prompts 20

uv run --env-file .env python -m llmtrain.judge \
    --input pairs_raw.jsonl \
    --output pairs_dpo.jsonl

uv run --env-file .env python -m llmtrain.training.dpo \
    --checkpoint my_sft_checkpoints_v2/step_12000.pt \
    --pairs pairs_dpo.jsonl \
    --checkpoint-dir dpo-pilot-checkpoints \
    --num-train-epochs 1
```

Expected: `generate_pairs.py` writes 20 rows to `pairs_raw.jsonl`; `judge.py` prints a summary (total/kept/discard_counts/mean_length_ratio) and writes the kept rows to `pairs_dpo.jsonl` — if `kept` is 0 or the discard rate looks implausibly high, that's the signal (per the spec) to revisit the judge prompt before scaling up, not to proceed; `training/dpo.py` runs to completion and writes a `step_N.pt` + `tokenizer.json` into `dpo-pilot-checkpoints/`.

- [ ] **Step 2: Confirm the exported checkpoint loads via the existing, unmodified `generate.py`**

```bash
uv run --env-file .env python -m llmtrain.generate \
    --checkpoint dpo-pilot-checkpoints/step_N.pt \
    --prompt "What is the capital of France?"
```

(substitute the actual `step_N.pt` filename `training/dpo.py` printed). Expected: loads without error and produces output — confirms the checkpoint-export step genuinely closed the loop back to existing tooling with zero changes to `generate.py` itself.

- [ ] **Step 3: Document the pipeline**

In `CLAUDE.md`'s Architecture file table (`src/llmtrain/` listing), add entries for `generate_pairs.py`, `judge.py`, `model/hf_wrapper.py`, and `training/dpo.py`, following the existing one-line-plus-wrapped-comment style already used for every other file in that table. In `docs/training-guide.md`, add a new part documenting the three-command pilot sequence from Step 1 above, the discard-rate signal from Step 1's expected output, and a note that `--num-prompts`/`num_train_epochs`/`--checkpoint-dir` should scale up to the spec's ~2,000-pair target once the pilot's discard rate and output quality look reasonable.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/training-guide.md
git commit -m "Document the DPO pipeline; validate end-to-end via a pilot run"
```
