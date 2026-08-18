import argparse
import json
import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import InferenceClient

from llmtrain.logging_config import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_PROVIDER = "together"
DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_JUDGE_TEMPERATURE = 0.15
_MAX_JUDGE_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 5.0

JUDGE_JSON_SCHEMA = {
    "name": "judge_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "verdict": {"type": "string", "enum": ["A", "B"]},
        },
        "required": ["reasoning", "verdict"],
        "additionalProperties": False,
    },
    "strict": True,
}

JUDGE_PROMPT_TEMPLATE = """You are evaluating two AI assistant responses to the same user prompt.

Guidelines:
- Judge only correctness, helpfulness, and relevance to the prompt.
- Do NOT prefer a response merely because it is longer or more detailed. A shorter, correct,
  on-point answer beats a longer one that pads with unnecessary detail or repetition. For
  example, given "What is 2+2?", the response "4" is better than a paragraph explaining basic
  arithmetic before arriving at "4".
- You must pick exactly one response as better. Ties are not allowed -- if genuinely close,
  break the tie on correctness, then relevance, then clarity, in that order.
- First write 1-2 sentences of reasoning, then give your verdict.

<prompt>
{prompt}
</prompt>

<response_a>
{response_a}
</response_a>

<response_b>
{response_b}
</response_b>"""


class JudgeParseError(Exception):
    pass


def build_judge_messages(prompt: str, response_a: str, response_b: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": JUDGE_PROMPT_TEMPLATE.format(
                prompt=prompt, response_a=response_a, response_b=response_b
            ),
        }
    ]


def call_judge(
    client: InferenceClient,
    model: str,
    prompt: str,
    response_a: str,
    response_b: str,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
) -> dict:
    completion = client.chat.completions.create(
        model=model,
        messages=build_judge_messages(prompt, response_a, response_b),
        response_format={"type": "json_schema", "json_schema": JUDGE_JSON_SCHEMA},
        temperature=temperature,
    )
    content = completion.choices[0].message.content
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JudgeParseError(f"judge response was not valid JSON: {content!r}") from exc
    if parsed.get("verdict") not in ("A", "B"):
        raise JudgeParseError(f"judge response missing/invalid 'verdict': {parsed!r}")
    return parsed


def call_judge_with_retry(
    client: InferenceClient,
    model: str,
    prompt: str,
    response_a: str,
    response_b: str,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    max_attempts: int = _MAX_JUDGE_ATTEMPTS,
    retry_delay: float = _RETRY_DELAY_SECONDS,
) -> dict | None:
    for attempt in range(1, max_attempts + 1):
        try:
            return call_judge(client, model, prompt, response_a, response_b, temperature)
        except JudgeParseError:
            # Not transient -- a malformed/off-schema response won't fix itself on retry.
            raise
        except Exception:
            if attempt == max_attempts:
                return None
            logger.warning(
                "judge call failed on attempt %d/%d, retrying",
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(retry_delay)
    return None


@dataclass
class JudgeResult:
    kept: bool
    chosen: str | None = None
    rejected: str | None = None
    discard_reason: str | None = None
    length_ratio: float | None = None


def judge_pair(
    client: InferenceClient,
    model: str,
    prompt: str,
    completion_a: str,
    completion_b: str,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    max_attempts: int = _MAX_JUDGE_ATTEMPTS,
    retry_delay: float = _RETRY_DELAY_SECONDS,
) -> JudgeResult:
    if completion_a == completion_b or not completion_a.strip() or not completion_b.strip():
        return JudgeResult(kept=False, discard_reason="degenerate_pair")
    try:
        forward = call_judge_with_retry(
            client, model, prompt, completion_a, completion_b, temperature, max_attempts, retry_delay
        )
        if forward is None:
            return JudgeResult(kept=False, discard_reason="api_failure")
        swapped = call_judge_with_retry(
            client, model, prompt, completion_b, completion_a, temperature, max_attempts, retry_delay
        )
        if swapped is None:
            return JudgeResult(kept=False, discard_reason="api_failure")
    except JudgeParseError:
        return JudgeResult(kept=False, discard_reason="parse_failure")

    # forward: A=completion_a, B=completion_b -> map the verdict back to which original
    # completion won. swapped: A=completion_b, B=completion_a -> same mapping, reversed.
    forward_winner = completion_a if forward["verdict"] == "A" else completion_b
    swapped_winner = completion_b if swapped["verdict"] == "A" else completion_a

    if forward_winner != swapped_winner:
        return JudgeResult(kept=False, discard_reason="position_bias_disagreement")

    winner = forward_winner
    loser = completion_b if winner == completion_a else completion_a
    # Character length, not token length -- a cheap diagnostic, not precise, and judge.py
    # has no reason to depend on this repo's tokenizer.
    length_ratio = len(winner) / max(len(loser), 1)
    return JudgeResult(kept=True, chosen=winner, rejected=loser, length_ratio=length_ratio)


def run_judge_pipeline(
    client: InferenceClient,
    model: str,
    rows: list[dict],
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    max_attempts: int = _MAX_JUDGE_ATTEMPTS,
    retry_delay: float = _RETRY_DELAY_SECONDS,
    output_path: str | Path | None = None,
) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    discard_counts = {
        "position_bias_disagreement": 0,
        "parse_failure": 0,
        "api_failure": 0,
        "degenerate_pair": 0,
    }
    length_ratios: list[float] = []
    with ExitStack() as stack:
        output_file = (
            stack.enter_context(open(output_path, "w")) if output_path is not None else None
        )
        for i, row in enumerate(rows, start=1):
            result = judge_pair(
                client,
                model,
                row["prompt"],
                row["completion_a"],
                row["completion_b"],
                temperature,
                max_attempts,
                retry_delay,
            )
            if result.kept:
                kept_row = {
                    "prompt": row["prompt"],
                    "chosen": result.chosen,
                    "rejected": result.rejected,
                }
                kept.append(kept_row)
                assert result.length_ratio is not None
                length_ratios.append(result.length_ratio)
                if output_file is not None:
                    output_file.write(json.dumps(kept_row) + "\n")
                    output_file.flush()
            else:
                assert result.discard_reason is not None
                discard_counts[result.discard_reason] += 1

            if i % 50 == 0:
                logger.info("judge pipeline progress: %d/%d processed, %d kept", i, len(rows), len(kept))
    summary = {
        "total": len(rows),
        "kept": len(kept),
        "discard_counts": discard_counts,
        "mean_length_ratio": sum(length_ratios) / len(length_ratios) if length_ratios else None,
    }
    return kept, summary


def _startup_self_check(client: InferenceClient, model: str, temperature: float) -> None:
    result = call_judge_with_retry(
        client,
        model,
        "What is 2+2?",
        "4",
        "The answer is definitely 5, I am very confident.",
        temperature,
    )
    if result is None:
        raise RuntimeError(
            f"judge startup self-check failed: no valid structured-output response from "
            f"provider/model {model!r}. Try a different --judge-provider/--judge-model pair."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge sampled completion pairs into DPO preference pairs"
    )
    parser.add_argument("--input", type=str, required=True, help="path to pairs_raw.jsonl")
    parser.add_argument("--output", type=str, required=True, help="path to write pairs_dpo.jsonl")
    parser.add_argument("--judge-provider", type=str, default=DEFAULT_JUDGE_PROVIDER)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_JUDGE_TEMPERATURE)
    parser.add_argument("--log-file", type=str, default="app.log")
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. Set it in .env and run with "
            "`uv run --env-file .env ...`."
        )
    client = InferenceClient(provider=args.judge_provider, api_key=hf_token)
    _startup_self_check(client, args.judge_model, args.temperature)

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    _kept, summary = run_judge_pipeline(
        client, args.judge_model, rows, args.temperature, output_path=args.output
    )

    logger.info("judge pipeline complete", extra=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
