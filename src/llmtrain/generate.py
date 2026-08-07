import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llmtrain.model.cache import KVCache
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.checkpoint import load_checkpoint
from llmtrain.training.config import GenerationConfig, ModelConfig


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float
) -> torch.Tensor:
    # Keskar et al. (CTRL, 2019) formula, also used by HF transformers: dividing a
    # positive logit (and multiplying a negative one) both push the token's probability
    # down, so the same `penalty >= 1.0` value works regardless of the logit's sign.
    if penalty == 1.0 or not generated_ids:
        return logits
    seen = torch.tensor(sorted(set(generated_ids)), dtype=torch.long, device=logits.device)
    seen_logits = logits[..., seen]
    seen_logits = torch.where(seen_logits < 0, seen_logits * penalty, seen_logits / penalty)
    logits = logits.clone()
    logits[..., seen] = seen_logits
    return logits


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative probability *up to and excluding* them already
    # exceeds top_p — the shift keeps the token that first crosses the threshold, so the
    # kept set's cumulative probability is always >= top_p (standard nucleus sampling).
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def _sample(
    logits: torch.Tensor,
    generated_ids: list[int],
    temperature: float,
    repetition_penalty: float,
    top_k: int,
    top_p: float,
) -> int:
    # Repetition penalty applies to greedy decoding too (it's what lets greedy avoid
    # repetition loops); top-k/top-p only affect the sampling distribution below.
    logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / temperature
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_token_ids(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: GenerationConfig,
) -> list[int]:
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("prompt encoded to zero tokens")
    if config.max_new_tokens <= 0:
        return prompt_ids

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        cache = KVCache()
        generated_ids = list(prompt_ids)

        def sample_next(logits: torch.Tensor) -> int:
            return _sample(
                logits,
                generated_ids,
                config.temperature,
                config.repetition_penalty,
                config.top_k,
                config.top_p,
            )

        with torch.no_grad():
            logits = model(input_ids, cache=cache)
            next_id = sample_next(logits[:, -1, :])
            generated_ids.append(next_id)
            for _ in range(config.max_new_tokens - 1):
                step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits = model(step_input, cache=cache)
                next_id = sample_next(logits[:, -1, :])
                generated_ids.append(next_id)
    finally:
        model.train(was_training)

    return generated_ids


def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: GenerationConfig,
) -> str:
    token_ids = generate_token_ids(model, tokenizer, prompt, config)
    return tokenizer.decode(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=GenerationConfig.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=GenerationConfig.temperature)
    parser.add_argument(
        "--repetition-penalty", type=float, default=GenerationConfig.repetition_penalty
    )
    parser.add_argument("--top-k", type=int, default=GenerationConfig.top_k)
    parser.add_argument("--top-p", type=float, default=GenerationConfig.top_p)
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
    model = TransformerLM(model_cfg)
    load_checkpoint(checkpoint_path, model)

    generation_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    output = generate(model, tokenizer, args.prompt, generation_cfg)
    print(output)


if __name__ == "__main__":
    main()
