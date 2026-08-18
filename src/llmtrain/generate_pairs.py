import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from llmtrain.data.chat import format_turn
from llmtrain.generate import generate_token_ids
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import TransformerLM
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import load_checkpoint
from llmtrain.training.config import GenerationConfig, ModelConfig

PROMPT_DATASET = "trl-lib/ultrafeedback-prompt"


def format_prompt(question: str) -> str:
    return format_turn("user", question) + "<|assistant|>\n"


def sample_completion(
    model: TransformerLM, tokenizer: Tokenizer, question: str, config: GenerationConfig
) -> str:
    prompt = format_prompt(question)
    prompt_ids = tokenizer.encode(prompt).ids
    full_ids = generate_token_ids(model, tokenizer, prompt, config)
    return tokenizer.decode(full_ids[len(prompt_ids) :])


def generate_pairs(
    model: TransformerLM,
    tokenizer: Tokenizer,
    questions: list[str],
    config: GenerationConfig,
) -> list[dict]:
    rows = []
    for question in questions:
        completion_a = sample_completion(model, tokenizer, question, config)
        completion_b = sample_completion(model, tokenizer, question, config)
        rows.append(
            {"prompt": question, "completion_a": completion_a, "completion_b": completion_b}
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample completion pairs for DPO judging")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--output", type=str, required=True, help="path to write pairs_raw.jsonl")
    parser.add_argument("--num-prompts", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=GenerationConfig.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=GenerationConfig.temperature)
    parser.add_argument(
        "--repetition-penalty", type=float, default=GenerationConfig.repetition_penalty
    )
    parser.add_argument("--top-k", type=int, default=GenerationConfig.top_k)
    parser.add_argument("--top-p", type=float, default=GenerationConfig.top_p)
    parser.add_argument("--log-file", type=str, default="app.log")
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)

    checkpoint_path = resolve_local_path(args.checkpoint)
    tokenizer_uri = args.tokenizer_path or sibling_path(args.checkpoint, "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(resolve_local_path(tokenizer_uri)))

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_model_config = raw_checkpoint.get("model_config")
    model_cfg = (
        ModelConfig(**{**saved_model_config, "vocab_size": tokenizer.get_vocab_size()})
        if saved_model_config is not None
        else ModelConfig(vocab_size=tokenizer.get_vocab_size())
    )
    del raw_checkpoint

    model = TransformerLM(model_cfg)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    dataset = load_dataset(PROMPT_DATASET, split="train", streaming=True)
    questions = [row["prompt"][0]["content"] for row in dataset.take(args.num_prompts)]

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    rows = generate_pairs(model, tokenizer, questions, config)

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
