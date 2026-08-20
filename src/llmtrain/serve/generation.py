from collections.abc import Iterator

import torch
from tokenizers import Tokenizer

from llmtrain.data.chat import format_chat_history
from llmtrain.data.tokenizer import PAD_TOKEN
from llmtrain.generate import _sample
from llmtrain.model.cache import KVCache
from llmtrain.model.transformer import TransformerLM
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.config import GenerationConfig, ModelConfig
from llmtrain.training.train import select_device

_VALID_ROLES = ("user", "assistant")

# Generous-but-bounded limits on the raw request shape, checked before any tokenization
# happens. Without these, a client can send an enormous `content` string or a huge
# `messages` list and truncate_to_context_window() will tokenize the whole thing --
# potentially more than once, once per truncation iteration -- before eventually
# rejecting it. That's real CPU-amplification exposure on a public endpoint, so it's
# checked here, before any tokenizer call.
MAX_MESSAGE_CONTENT_CHARS = 8000  # generous for a single chat message
MAX_MESSAGE_COUNT = 50  # generous for a chat history


def validate_messages(messages: list[dict] | None) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    if len(messages) > MAX_MESSAGE_COUNT:
        raise ValueError(
            f"messages has {len(messages)} entries, exceeds limit of {MAX_MESSAGE_COUNT}"
        )

    expected_role = "user"
    for i, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role not in _VALID_ROLES:
            raise ValueError(
                f"message {i} has invalid role {role!r}; must be 'user' or 'assistant'"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(f"message {i} must have non-empty string 'content'")
        if len(content) > MAX_MESSAGE_CONTENT_CHARS:
            raise ValueError(
                f"message {i} content is {len(content)} chars, exceeds limit of "
                f"{MAX_MESSAGE_CONTENT_CHARS}"
            )
        if role != expected_role:
            raise ValueError(
                f"messages must alternate starting with 'user'; message {i} has role "
                f"{role!r}, expected {expected_role!r}"
            )
        expected_role = "assistant" if expected_role == "user" else "user"

    if messages[-1]["role"] != "user":
        raise ValueError("messages must end on a 'user' turn")


MAX_NEW_TOKENS_CEILING = 512


def parse_generation_config(payload: dict) -> GenerationConfig:
    requested_max_new_tokens = int(payload.get("max_new_tokens", GenerationConfig.max_new_tokens))
    temperature = float(payload.get("temperature", GenerationConfig.temperature))
    top_p = float(payload.get("top_p", GenerationConfig.top_p))
    top_k = int(payload.get("top_k", GenerationConfig.top_k))
    repetition_penalty = float(
        payload.get("repetition_penalty", GenerationConfig.repetition_penalty)
    )
    return GenerationConfig(
        max_new_tokens=min(requested_max_new_tokens, MAX_NEW_TOKENS_CEILING),
        # 0.0 is greedy decoding, an existing meaningful value in _sample() -- clamp the
        # top end only, don't push 0.0 up to some minimum.
        temperature=max(0.0, min(temperature, 2.0)),
        # <= 0.0 has no sane "smallest valid value" to clamp to, so it falls back to 1.0
        # (top_p disabled/no-op) rather than some tiny positive epsilon.
        top_p=1.0 if top_p <= 0.0 else min(top_p, 1.0),
        top_k=max(0, top_k),
        # 1.0 is the existing no-op value (see generate.py's _apply_repetition_penalty).
        repetition_penalty=max(1.0, min(repetition_penalty, 2.0)),
    )


def truncate_to_context_window(
    tokenizer: Tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    max_seq_len: int,
) -> list[dict]:
    messages = list(messages)
    while True:
        prompt_len = len(tokenizer.encode(format_chat_history(messages)).ids)
        if prompt_len + max_new_tokens <= max_seq_len:
            return messages
        if len(messages) <= 1:
            raise ValueError("prompt exceeds max_seq_len even after truncating all prior turns")
        messages = messages[2:]  # drop the oldest user/assistant turn-pair, never mid-turn


def load_model_and_tokenizer(
    checkpoint_path: str, tokenizer_path: str | None = None
) -> tuple[TransformerLM, Tokenizer]:
    tokenizer_uri = tokenizer_path or sibling_path(checkpoint_path, "tokenizer.json")
    resolved_checkpoint = resolve_local_path(checkpoint_path)
    resolved_tokenizer = resolve_local_path(tokenizer_uri)

    tokenizer = Tokenizer.from_file(str(resolved_tokenizer))

    # A single torch.load() here, rather than calling both
    # load_model_config_from_checkpoint() and load_checkpoint() (each of which does its
    # own full torch.load() of the same file), halves the number of complete
    # deserializations of a multi-GB checkpoint off a network-mounted volume on every
    # cold start -- cold start is the dominant user-visible latency for this endpoint.
    # This inlines both helpers' exact logic rather than changing checkpoint.py itself:
    # that module is shared by train.py/generate.py/dpo.py, and widening its API for
    # this one caller isn't worth the regression risk for a small amount of local
    # duplication (see CLAUDE.md's "Karpathy principles for overengineering").
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu")
    saved_model_config = checkpoint.get("model_config")
    if saved_model_config is not None:
        model_cfg = ModelConfig(**{**saved_model_config, "vocab_size": tokenizer.get_vocab_size()})
    else:
        model_cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = TransformerLM(model_cfg)
    model.load_state_dict(checkpoint["model_state"])

    # The deployed serverless worker runs on a real GPU; without this the model stays
    # on the CPU default TransformerLM(model_cfg) constructs it with, and every request
    # silently runs CPU inference despite the GPU being provisioned and paid for.
    device = select_device()
    model.to(device)
    model.eval()
    return model, tokenizer


def stream_chat_completion(
    model: TransformerLM,
    tokenizer: Tokenizer,
    # list[dict] | None (not just list[dict]) because handler.py passes a job
    # payload's `messages` field through untouched -- a client can omit it entirely,
    # and validate_messages() (called first below) is what turns that None into a
    # clean ValueError rather than a type error.
    messages: list[dict] | None,
    generation_cfg: GenerationConfig,
    max_seq_len: int,
) -> Iterator[str]:
    validate_messages(messages)
    # validate_messages() above already raises ValueError for a None/malformed
    # `messages`, so this is a pure type-narrowing assertion for mypy (list[dict] |
    # None -> list[dict]) below, not a new runtime check.
    assert messages is not None
    messages = truncate_to_context_window(
        tokenizer, messages, generation_cfg.max_new_tokens, max_seq_len
    )
    prompt_ids = tokenizer.encode(format_chat_history(messages)).ids
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    if generation_cfg.max_new_tokens <= 0:
        return

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        cache = KVCache(max_seq_len=len(prompt_ids) + generation_cfg.max_new_tokens)
        generated_ids = list(prompt_ids)
        new_ids: list[int] = []
        prev_text = ""

        def sample_next(logits: torch.Tensor) -> int:
            return _sample(
                logits,
                generated_ids,
                generation_cfg.temperature,
                generation_cfg.repetition_penalty,
                generation_cfg.top_k,
                generation_cfg.top_p,
            )

        def emit(token_id: int) -> str:
            nonlocal prev_text
            new_ids.append(token_id)
            generated_ids.append(token_id)
            # skip_special_tokens=False: PAD_TOKEN is provably never appended here (the
            # decode loop below breaks before emitting it), so that part is a true
            # no-op. But nothing stops the model from *sampling* UNK (id 0) as an
            # output token in principle -- unlikely for a real trained model, but not
            # impossible. If it happened, skip_special_tokens=False means it surfaces
            # literally as the string "[UNK]" in the stream, rather than being silently
            # decoded to '' and dropped -- arguably the more honest behavior for a
            # public-facing API (a visible artifact beats invisible corruption), not
            # purely a no-op like the PAD_TOKEN case.
            full_text = tokenizer.decode(new_ids, skip_special_tokens=False)
            # Byte-level BPE can split a multi-byte UTF-8 character across two token
            # ids; decoding a still-incomplete sequence produces a trailing U+FFFD
            # replacement character. If we advanced prev_text and yielded here, that
            # "�" would already be on its way to the client, and the *next* decode
            # call (once the sequence resolves) produces the same *length* of text (one
            # replacement char -> one real char), so the diff-based delta would come out
            # empty and the corrupted "�" already sent would never get corrected.
            # Holding back here (not advancing prev_text, yielding nothing) means the
            # pending bytes are naturally caught up in the delta on the step where the
            # sequence resolves, since full_text is always the full decode of every
            # token so far.
            if full_text.endswith("�"):
                return ""
            delta = full_text[len(prev_text) :]
            prev_text = full_text
            return delta

        with torch.no_grad():
            logits = model(input_ids, cache=cache)
            next_id = sample_next(logits[:, -1, :])
            if next_id != pad_id:
                delta = emit(next_id)
                if delta:
                    yield delta
                for _ in range(generation_cfg.max_new_tokens - 1):
                    step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    logits = model(step_input, cache=cache)
                    next_id = sample_next(logits[:, -1, :])
                    if next_id == pad_id:
                        break
                    delta = emit(next_id)
                    if delta:
                        yield delta
    finally:
        model.train(was_training)
