import json

import pytest

from llmtrain.judge import (
    JudgeParseError,
    call_judge,
    call_judge_with_retry,
    judge_pair,
    run_judge_pipeline,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses: list) -> None:
        self._responses = iter(responses)

    def create(self, **kwargs):
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return _FakeCompletion(response)


class _FakeChat:
    def __init__(self, responses: list) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.chat = _FakeChat(responses)


def _verdict(verdict: str, reasoning: str = "because") -> str:
    return json.dumps({"reasoning": reasoning, "verdict": verdict})


def test_call_judge_parses_a_valid_response():
    client = _FakeClient([_verdict("A")])
    result = call_judge(client, "some-model", "prompt", "resp a", "resp b")
    assert result["verdict"] == "A"


def test_call_judge_raises_judge_parse_error_on_malformed_json():
    client = _FakeClient(["not json"])
    with pytest.raises(JudgeParseError):
        call_judge(client, "some-model", "prompt", "resp a", "resp b")


def test_call_judge_raises_judge_parse_error_on_missing_verdict():
    client = _FakeClient([json.dumps({"reasoning": "because"})])
    with pytest.raises(JudgeParseError):
        call_judge(client, "some-model", "prompt", "resp a", "resp b")


def test_call_judge_with_retry_recovers_from_a_transient_failure():
    client = _FakeClient([RuntimeError("timeout"), _verdict("B")])
    result = call_judge_with_retry(
        client, "some-model", "prompt", "resp a", "resp b", max_attempts=3, retry_delay=0.0
    )
    assert result["verdict"] == "B"


def test_call_judge_with_retry_gives_up_after_max_attempts():
    client = _FakeClient([RuntimeError("timeout")] * 3)
    result = call_judge_with_retry(
        client, "some-model", "prompt", "resp a", "resp b", max_attempts=3, retry_delay=0.0
    )
    assert result is None


def test_judge_pair_keeps_the_pair_when_both_orderings_agree():
    # forward call: A=completion_a, B=completion_b -> "A" means completion_a won
    # swapped call: A=completion_b, B=completion_a -> "B" means completion_a won again
    client = _FakeClient([_verdict("A"), _verdict("B")])
    result = judge_pair(client, "some-model", "prompt", "completion_a text", "completion_b text")
    assert result.kept
    assert result.chosen == "completion_a text"
    assert result.rejected == "completion_b text"
    assert result.length_ratio is not None


def test_judge_pair_discards_on_position_bias_disagreement():
    # forward call: A=completion_a, B=completion_b -> "A" means completion_a won
    # swapped call: A=completion_b, B=completion_a -> "A" means completion_b won this time
    client = _FakeClient([_verdict("A"), _verdict("A")])
    result = judge_pair(client, "some-model", "prompt", "completion_a text", "completion_b text")
    assert not result.kept
    assert result.discard_reason == "position_bias_disagreement"


def test_judge_pair_discards_on_parse_failure():
    client = _FakeClient(["not json", _verdict("A")])
    result = judge_pair(client, "some-model", "prompt", "completion_a text", "completion_b text")
    assert not result.kept
    assert result.discard_reason == "parse_failure"


def test_judge_pair_discards_on_api_failure():
    # Only 3 RuntimeErrors queued (not 6): the forward call exhausts its max_attempts=3
    # retries and judge_pair short-circuits before ever attempting the swapped call.
    client = _FakeClient([RuntimeError("timeout")] * 3)
    result = judge_pair(
        client,
        "some-model",
        "prompt",
        "completion_a text",
        "completion_b text",
        retry_delay=0.0,
    )
    assert not result.kept
    assert result.discard_reason == "api_failure"


def test_judge_pair_discards_degenerate_pairs_without_calling_the_judge():
    # Zero responses queued -- any call to .create() would raise StopIteration, proving
    # the judge is never actually invoked for a degenerate (identical) pair.
    client = _FakeClient([])
    result = judge_pair(client, "some-model", "prompt", "same text", "same text")
    assert not result.kept
    assert result.discard_reason == "degenerate_pair"


def test_judge_pair_discards_blank_completions_without_calling_the_judge():
    client = _FakeClient([])
    result = judge_pair(client, "some-model", "prompt", "", "")
    assert not result.kept
    assert result.discard_reason == "degenerate_pair"


def test_run_judge_pipeline_tallies_kept_and_discard_counts():
    # row 1: agree -> kept. row 2: disagree -> position_bias_disagreement.
    client = _FakeClient([_verdict("A"), _verdict("B"), _verdict("A"), _verdict("A")])
    rows = [
        {"prompt": "p1", "completion_a": "a1", "completion_b": "b1"},
        {"prompt": "p2", "completion_a": "a2", "completion_b": "b2"},
    ]
    kept, summary = run_judge_pipeline(client, "some-model", rows)
    assert summary["total"] == 2
    assert summary["kept"] == 1
    assert summary["discard_counts"]["position_bias_disagreement"] == 1
    assert kept == [{"prompt": "p1", "chosen": "a1", "rejected": "b1"}]


def test_run_judge_pipeline_writes_kept_rows_incrementally_to_output_path(tmp_path):
    client = _FakeClient([_verdict("A"), _verdict("B"), _verdict("A"), _verdict("A")])
    rows = [
        {"prompt": "p1", "completion_a": "a1", "completion_b": "b1"},
        {"prompt": "p2", "completion_a": "a2", "completion_b": "b2"},
    ]
    output_path = tmp_path / "out.jsonl"

    kept, _summary = run_judge_pipeline(client, "some-model", rows, output_path=output_path)

    written = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert written == kept
