import pytest

from llmtrain.serve import generation, handler


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


def test_handler_yields_structured_error_on_malformed_generation_config(monkeypatch):
    # parse_generation_config does bare int()/float() coercion on client-supplied
    # values; a payload like {"max_new_tokens": "abc"} raises ValueError/TypeError.
    # Before this fix, parse_generation_config was called before the try block, so
    # this propagated uncaught instead of becoming a structured error response.
    monkeypatch.setattr(handler, "_get_model_and_tokenizer", lambda: ("fake-model", "fake-tok"))

    job = {"input": {"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": "abc"}}
    chunks = list(handler.handler(job))

    assert len(chunks) == 1
    assert chunks[0]["done"] is True
    assert "error" in chunks[0]


@pytest.mark.parametrize("bad_input", ["not a dict", ["also", "not", "a", "dict"], None, 42])
def test_handler_treats_non_dict_input_as_missing_messages(monkeypatch, bad_input):
    # RunPod's job["input"] is client-controlled; a non-dict input used to hit
    # payload.get(...) directly and raise an uncaught AttributeError. It should
    # instead degrade to "no messages provided" and surface as a structured error.
    monkeypatch.setattr(handler, "_get_model_and_tokenizer", lambda: ("fake-model", "fake-tok"))

    job = {"input": bad_input}
    chunks = list(handler.handler(job))

    assert len(chunks) == 1
    assert chunks[0]["done"] is True
    assert "error" in chunks[0]


def test_handler_clamps_max_new_tokens_before_calling_stream_chat_completion(monkeypatch):
    monkeypatch.setattr(handler, "_get_model_and_tokenizer", lambda: ("fake-model", "fake-tok"))

    captured_calls = []

    def fake_stream_chat_completion(model, tokenizer, messages, generation_cfg, max_seq_len):
        captured_calls.append(generation_cfg)
        return iter([])

    monkeypatch.setattr(handler.generation, "stream_chat_completion", fake_stream_chat_completion)

    requested_max_new_tokens = generation.MAX_NEW_TOKENS_CEILING + 1000
    job = {
        "input": {
            "messages": [{"role": "user", "content": "hi"}],
            "max_new_tokens": requested_max_new_tokens,
        }
    }
    list(handler.handler(job))

    assert len(captured_calls) == 1
    # The handler must pass the *clamped* value through to stream_chat_completion --
    # not the raw client-requested value -- since max_new_tokens is the main
    # cost-control property the whole API design depends on.
    assert captured_calls[0].max_new_tokens == generation.MAX_NEW_TOKENS_CEILING
    assert captured_calls[0].max_new_tokens != requested_max_new_tokens
