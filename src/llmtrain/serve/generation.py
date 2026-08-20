from collections.abc import Iterator

import torch
from tokenizers import Tokenizer

from llmtrain.data.chat import format_chat_history
from llmtrain.data.tokenizer import PAD_TOKEN
from llmtrain.generate import _sample
from llmtrain.model.cache import KVCache
from llmtrain.model.transformer import TransformerLM
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import load_checkpoint, load_model_config_from_checkpoint
from llmtrain.training.config import GenerationConfig

_VALID_ROLES = ("user", "assistant")


def validate_messages(messages: list[dict] | None) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

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
    return GenerationConfig(
        max_new_tokens=min(requested_max_new_tokens, MAX_NEW_TOKENS_CEILING),
        temperature=float(payload.get("temperature", GenerationConfig.temperature)),
        repetition_penalty=float(
            payload.get("repetition_penalty", GenerationConfig.repetition_penalty)
        ),
        top_k=int(payload.get("top_k", GenerationConfig.top_k)),
        top_p=float(payload.get("top_p", GenerationConfig.top_p)),
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
    model_cfg = load_model_config_from_checkpoint(resolved_checkpoint, tokenizer.get_vocab_size())
    model = TransformerLM(model_cfg)
    load_checkpoint(resolved_checkpoint, model)
    model.eval()
    return model, tokenizer


def stream_chat_completion(
    model: TransformerLM,
    tokenizer: Tokenizer,
    messages: list[dict],
    generation_cfg: GenerationConfig,
    max_seq_len: int,
) -> Iterator[str]:
    validate_messages(messages)
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
            # skip_special_tokens=False: PAD_TOKEN is never appended here (the decode
            # loop breaks before emitting it) and UNK never fires for a real trained
            # tokenizer (byte-level BPE covers every input byte -- see data/tokenizer.py),
            # so this only matters for a test stub that deliberately emits a special-token
            # id as a stand-in "real" token; using the default skip_special_tokens=True
            # would silently decode such an id to '', breaking the decode-diff.
            full_text = tokenizer.decode(new_ids, skip_special_tokens=False)
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
