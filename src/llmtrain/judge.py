import argparse
import json
import logging
import os
import time
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

Prompt: {prompt}
Response A: {response_a}
Response B: {response_b}"""


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
) -> JudgeResult:
    try:
        forward = call_judge_with_retry(client, model, prompt, completion_a, completion_b, temperature)
        swapped = call_judge_with_retry(client, model, prompt, completion_b, completion_a, temperature)
    except JudgeParseError:
        return JudgeResult(kept=False, discard_reason="parse_failure")

    if forward is None or swapped is None:
        return JudgeResult(kept=False, discard_reason="api_failure")

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
) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    discard_counts = {"position_bias_disagreement": 0, "parse_failure": 0, "api_failure": 0}
    length_ratios: list[float] = []
    for row in rows:
        result = judge_pair(
            client, model, row["prompt"], row["completion_a"], row["completion_b"], temperature
        )
        if result.kept:
            kept.append({"prompt": row["prompt"], "chosen": result.chosen, "rejected": result.rejected})
            assert result.length_ratio is not None
            length_ratios.append(result.length_ratio)
        else:
            assert result.discard_reason is not None
            discard_counts[result.discard_reason] += 1
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
    client = InferenceClient(provider=args.judge_provider, api_key=os.environ["HF_TOKEN"])
    _startup_self_check(client, args.judge_model, args.temperature)

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    kept, summary = run_judge_pipeline(client, args.judge_model, rows, args.temperature)

    with open(args.output, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in kept)

    logger.info("judge pipeline complete", extra=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
