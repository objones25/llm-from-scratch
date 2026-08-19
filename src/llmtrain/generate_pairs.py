import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from llmtrain.data.chat import format_prompt
from llmtrain.generate import generate_token_ids
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import TransformerLM
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import load_checkpoint, load_model_config_from_checkpoint
from llmtrain.training.config import GenerationConfig, TrainConfig
from llmtrain.training.train import select_device

logger = logging.getLogger(__name__)

PROMPT_DATASET = "trl-lib/ultrafeedback-prompt"
_DEFAULT_PROGRESS_INTERVAL = 10


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
    output_path: str | Path | None = None,
    progress_interval: int = _DEFAULT_PROGRESS_INTERVAL,
    resume_from: int = 0,
) -> list[dict]:
    # Writes each row incrementally (mirroring judge.py's run_judge_pipeline) so a
    # crash/kill mid-run leaves a usable partial pairs_raw.jsonl instead of losing every
    # completed generation -- this loop can run for hours unattended on a rented GPU.
    # resume_from skips questions already generated in a prior (interrupted) run --
    # unlike judge.py, no separate progress marker is needed: every question here
    # produces exactly one written row, so "rows already in --output" directly gives the
    # resume point (main() computes it that way). output_path opens in append mode when
    # resuming so already-written rows survive instead of being truncated.
    rows: list[dict] = []
    with ExitStack() as stack:
        output_file = (
            stack.enter_context(open(output_path, "a" if resume_from else "w"))
            if output_path is not None
            else None
        )
        for i, question in enumerate(questions[resume_from:], start=resume_from + 1):
            completion_a = sample_completion(model, tokenizer, question, config)
            completion_b = sample_completion(model, tokenizer, question, config)
            row = {
                "prompt": question,
                "completion_a": completion_a,
                "completion_b": completion_b,
            }
            rows.append(row)
            if output_file is not None:
                output_file.write(json.dumps(row) + "\n")
                output_file.flush()
            if i % progress_interval == 0:
                logger.info(
                    "generate_pairs progress: %d/%d prompts, %d pairs generated",
                    i,
                    len(questions),
                    len(rows),
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
    parser.add_argument("--log-file", type=str, default=TrainConfig.log_file)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume an interrupted run: skip questions already generated (counted from "
            "existing rows in --output) and append to --output instead of overwriting it. "
            "Requires the same --num-prompts as the interrupted run, since the resumed "
            "question list must line up with what's already on disk."
        ),
    )
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)

    checkpoint_path = resolve_local_path(args.checkpoint)
    tokenizer_uri = args.tokenizer_path or sibling_path(args.checkpoint, "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(resolve_local_path(tokenizer_uri)))

    model_cfg = load_model_config_from_checkpoint(checkpoint_path, tokenizer.get_vocab_size())
    model = TransformerLM(model_cfg)
    load_checkpoint(checkpoint_path, model)
    device = select_device()
    model.to(device)
    model.eval()

    dataset = load_dataset(PROMPT_DATASET, split="train", streaming=True)
    questions = [row["prompt"][0]["content"] for row in dataset.take(args.num_prompts)]

    resume_from = 0
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        resume_from = sum(1 for line in output_path.read_text().splitlines() if line.strip())
        logger.info("resuming generate_pairs from prompt %d/%d", resume_from, len(questions))

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    generate_pairs(model, tokenizer, questions, config, output_path=args.output, resume_from=resume_from)


if __name__ == "__main__":
    main()
