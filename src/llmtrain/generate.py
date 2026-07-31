import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llmtrain.model.cache import KVCache
from llmtrain.model.transformer import MinimalTransformerLM
from llmtrain.training.checkpoint import load_checkpoint
from llmtrain.training.config import ModelConfig


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
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    if max_new_tokens <= 0:
        return prompt_ids

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
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
    finally:
        model.train(was_training)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = (
        Path(args.tokenizer_path)
        if args.tokenizer_path
        else checkpoint_path.parent / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Peek the checkpoint for its persisted model_config so architecture fields that change
    # numerics without changing tensor shapes (e.g. rope_theta) can't silently drift from
    # what the checkpoint was actually trained with. Older checkpoints saved before
    # model_config was persisted fall back to ModelConfig() defaults, as before.
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_model_config = raw_checkpoint.get("model_config")
    if saved_model_config is not None:
        model_cfg = ModelConfig(**{**saved_model_config, "vocab_size": tokenizer.get_vocab_size()})
    else:
        model_cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = MinimalTransformerLM(model_cfg)
    # load_checkpoint requires an optimizer arg; inference discards it.
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    load_checkpoint(checkpoint_path, model, dummy_optimizer)

    output = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
    print(output)


if __name__ == "__main__":
    main()
