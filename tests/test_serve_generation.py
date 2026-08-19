import pytest

from llmtrain.serve.generation import (
    MAX_NEW_TOKENS_CEILING,
    parse_generation_config,
    validate_messages,
)
from llmtrain.training.config import GenerationConfig


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
