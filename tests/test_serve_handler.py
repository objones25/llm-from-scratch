import pytest

from llmtrain.serve import handler


@pytest.fixture(autouse=True)
def _reset_model_cache():
    handler._model = None
    handler._tokenizer = None
    yield
    handler._model = None
    handler._tokenizer = None


def test_get_model_and_tokenizer_loads_once_and_caches(monkeypatch):
    calls = []

    def fake_load(checkpoint_path, tokenizer_path):
        calls.append((checkpoint_path, tokenizer_path))
        return "fake-model", "fake-tokenizer"

    monkeypatch.setattr(handler.generation, "load_model_and_tokenizer", fake_load)

    first = handler._get_model_and_tokenizer()
    second = handler._get_model_and_tokenizer()

    assert first == ("fake-model", "fake-tokenizer")
    assert second == first
    assert len(calls) == 1


def test_handler_yields_streamed_tokens_then_done(monkeypatch):
    monkeypatch.setattr(handler, "_get_model_and_tokenizer", lambda: ("fake-model", "fake-tok"))
    monkeypatch.setattr(
        handler.generation, "stream_chat_completion", lambda *a, **k: iter(["He", "llo"])
    )

    job = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    chunks = list(handler.handler(job))

    assert chunks == [
        {"token": "He", "done": False},
        {"token": "llo", "done": False},
        {"done": True},
    ]


def test_handler_yields_structured_error_on_invalid_input(monkeypatch):
    monkeypatch.setattr(handler, "_get_model_and_tokenizer", lambda: ("fake-model", "fake-tok"))

    def raise_value_error(*args, **kwargs):
        raise ValueError("messages must be a non-empty list")

    monkeypatch.setattr(handler.generation, "stream_chat_completion", raise_value_error)

    job = {"input": {"messages": []}}
    chunks = list(handler.handler(job))

    assert chunks == [{"error": "messages must be a non-empty list", "done": True}]
