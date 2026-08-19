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
