import torch
from tokenizers import Tokenizer

from llmtrain.model.cache import KVCache
from llmtrain.model.transformer import MinimalTransformerLM


def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_token_ids(
    model: MinimalTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> list[int]:
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    cache = KVCache()
    generated_ids = list(prompt_ids)
    with torch.no_grad():
        logits = model(input_ids, cache=cache)
        next_id = _sample(logits[:, -1, :], temperature)
        generated_ids.append(next_id)
        for _ in range(max_new_tokens - 1):
            step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(step_input, cache=cache)
            next_id = _sample(logits[:, -1, :], temperature)
            generated_ids.append(next_id)

    return generated_ids


def generate(
    model: MinimalTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> str:
    token_ids = generate_token_ids(model, tokenizer, prompt, max_new_tokens, temperature)
    return tokenizer.decode(token_ids)
