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
    if max_new_tokens <= 0:
        return prompt_ids

    model.eval()
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
        Path(args.tokenizer_path) if args.tokenizer_path else checkpoint_path.parent / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ModelConfig() defaults must match the training-time architecture. train.py's
    # CLI doesn't yet override architecture fields (only max_steps/batch_size/lr/
    # checkpoint_dir), so this holds today; a future config-rightsizing spec that
    # adds architecture CLI overrides to train.py must persist them for generate.py too.
    model_cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = MinimalTransformerLM(model_cfg)
    # load_checkpoint requires an optimizer arg; inference discards it.
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    load_checkpoint(checkpoint_path, model, dummy_optimizer)

    output = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
    print(output)


if __name__ == "__main__":
    main()
