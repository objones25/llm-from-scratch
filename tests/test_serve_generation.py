from unittest.mock import patch

import pytest
import torch

from llmtrain.data.chat import format_chat_history
from llmtrain.data.tokenizer import PAD_TOKEN, train_tokenizer
from llmtrain.generate import generate_token_ids
from llmtrain.model.transformer import TransformerLM
from llmtrain.serve.generation import (
    MAX_MESSAGE_CONTENT_CHARS,
    MAX_MESSAGE_COUNT,
    MAX_NEW_TOKENS_CEILING,
    load_model_and_tokenizer,
    parse_generation_config,
    stream_chat_completion,
    truncate_to_context_window,
    validate_messages,
)
from llmtrain.training.checkpoint import save_checkpoint
from llmtrain.training.config import GenerationConfig, ModelConfig
from llmtrain.training.train import select_device


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


def _alternating_messages(n: int) -> list[dict]:
    # A valid conversation must start on 'user', strictly alternate, and end on
    # 'user' -- which is only satisfiable for odd n. MAX_MESSAGE_COUNT (50) is even,
    # so there is no valid 50-message conversation to test the count check's exact
    # boundary with; the tests below instead use the largest valid (odd) length not
    # exceeding the limit (49) for the "accepted" case, and the smallest valid (odd)
    # length that does exceed it (51) for the "rejected" case.
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": f"msg {i}"} for i in range(n)]


def test_validate_messages_rejects_too_many_messages():
    messages = _alternating_messages(MAX_MESSAGE_COUNT + 1)
    assert len(messages) % 2 == 1  # ends on 'user'
    assert len(messages) > MAX_MESSAGE_COUNT
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_validate_messages_accepts_message_count_at_the_limit():
    messages = _alternating_messages(MAX_MESSAGE_COUNT - 1)
    assert len(messages) % 2 == 1  # ends on 'user'
    assert len(messages) <= MAX_MESSAGE_COUNT
    validate_messages(messages)  # no exception


def test_validate_messages_rejects_content_over_the_char_limit():
    with pytest.raises(ValueError):
        validate_messages([{"role": "user", "content": "x" * (MAX_MESSAGE_CONTENT_CHARS + 1)}])


def test_validate_messages_accepts_content_at_the_char_limit():
    validate_messages(
        [{"role": "user", "content": "x" * MAX_MESSAGE_CONTENT_CHARS}]
    )  # no exception


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


def test_parse_generation_config_clamps_temperature_to_valid_range():
    assert parse_generation_config({"temperature": -1.0}).temperature == 0.0
    assert parse_generation_config({"temperature": 5.0}).temperature == 2.0
    # 0.0 (greedy) is an existing meaningful value, not something to push upward.
    assert parse_generation_config({"temperature": 0.0}).temperature == 0.0


def test_parse_generation_config_clamps_top_p_to_valid_range():
    assert parse_generation_config({"top_p": 1.5}).top_p == 1.0
    # <= 0.0 has no sane smallest-positive-value fallback, so it disables top_p (1.0)
    # rather than clamping to some tiny epsilon.
    assert parse_generation_config({"top_p": 0.0}).top_p == 1.0
    assert parse_generation_config({"top_p": -1.0}).top_p == 1.0
    assert parse_generation_config({"top_p": 0.5}).top_p == 0.5


def test_parse_generation_config_clamps_negative_top_k_to_zero():
    assert parse_generation_config({"top_k": -5}).top_k == 0
    assert parse_generation_config({"top_k": 40}).top_k == 40


def test_parse_generation_config_clamps_repetition_penalty_to_valid_range():
    assert parse_generation_config({"repetition_penalty": 0.5}).repetition_penalty == 1.0
    assert parse_generation_config({"repetition_penalty": 10.0}).repetition_penalty == 2.0
    assert parse_generation_config({"repetition_penalty": 1.3}).repetition_penalty == 1.3


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
    # The deployed serverless worker runs on a real GPU; select_device() is the same
    # accelerator-detection call train.py uses, so on this dev machine it resolves to
    # whatever select_device() itself reports (mps/cpu here, cuda on a real worker) --
    # asserted against its own return value rather than a hardcoded device string so
    # this doesn't break on a machine with a different accelerator.
    expected_device = select_device()
    actual_device = next(loaded_model.parameters()).device
    assert actual_device.type == expected_device.type


def test_load_model_and_tokenizer_reads_the_checkpoint_file_exactly_once(tmp_path):
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

    # Regression guard for the cold-start double-deserialization finding: two full
    # torch.load() calls against the same multi-GB checkpoint file on every cold
    # start was the actual bug (load_model_config_from_checkpoint() +
    # load_checkpoint() each did their own torch.load()). Spying on torch.load itself
    # (rather than just checking the round-trip still works, which would pass either
    # way) pins "exactly once" as the real regression guard.
    with patch("llmtrain.serve.generation.torch.load", wraps=torch.load) as mock_load:
        load_model_and_tokenizer(str(checkpoint_path))

    assert mock_load.call_count == 1


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


class _FixedSequenceModel(torch.nn.Module):
    """Greedily emits a pre-scripted sequence of token ids, one per forward() call."""

    def __init__(self, vocab_size: int, token_ids: list[int]):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_ids = token_ids
        self.calls = 0
        self._unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, cache=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -10.0)
        logits[:, -1, self.token_ids[self.calls]] = 10.0
        self.calls += 1
        return logits

    def parameters(self, recurse=True):
        return iter([self._unused])


def test_stream_chat_completion_never_yields_unicode_replacement_character():
    # Byte-level BPE can split a multi-byte UTF-8 character across two token ids.
    # With a small vocab, "café" reliably reproduces this: confirmed empirically that
    # decoding the first 4 of its 5 token ids ends in U+FFFD (the second byte of "é"'s
    # 2-byte UTF-8 encoding hasn't arrived yet), and decoding all 5 resolves to "café"
    # cleanly. Scripting a fake model to emit exactly this token id sequence (rather
    # than hoping a real trained model happens to generate it) makes this a reliable,
    # non-flaky regression test for the decode-diff bug in emit() -- feeding raw token
    # ids confirmed via the real tokenizer to produce a replacement-character-then-
    # resolve pattern, per the finding's own suggested fallback approach.
    texts = [
        "hello world",
        "hello there",
        "I love a nice cup of cafe au lait",
        "café is a word",
        "the cafe sells café pastries",
    ]
    tokenizer = train_tokenizer(texts, vocab_size=64)
    token_ids = tokenizer.encode("café").ids
    assert tokenizer.decode(token_ids[:4], skip_special_tokens=False).endswith("�")
    assert tokenizer.decode(token_ids, skip_special_tokens=False) == "café"

    model = _FixedSequenceModel(vocab_size=tokenizer.get_vocab_size(), token_ids=token_ids)
    messages = [{"role": "user", "content": "hello"}]

    chunks = list(
        stream_chat_completion(
            model,
            tokenizer,
            messages,
            GenerationConfig(max_new_tokens=len(token_ids), temperature=0.0),
            max_seq_len=2048,
        )
    )

    assert all("�" not in chunk for chunk in chunks)
    assert "".join(chunks) == "café"


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
    tokenizer = train_tokenizer(["hello world", "hello there", "world hello there"], vocab_size=32)
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
