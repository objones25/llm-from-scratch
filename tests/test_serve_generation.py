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
