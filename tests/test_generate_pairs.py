import json
import logging

import torch
from tokenizers import Tokenizer

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.generate_pairs import format_prompt, generate_pairs, sample_completion
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import GenerationConfig, ModelConfig


def _tiny_setup() -> tuple[TransformerLM, Tokenizer]:
    tokenizer = train_tokenizer(
        ["<|user|>\nhello\n<|assistant|>\nhi there\n", "<|user|>\nbye\n<|assistant|>\nsee you\n"],
        vocab_size=64,
    )
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=8,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        dropout=0.0,
    )
    model = TransformerLM(config)
    return model, tokenizer


def test_format_prompt_wraps_question_in_user_and_assistant_tags():
    assert format_prompt("hello") == "<|user|>\nhello\n<|assistant|>\n"


def test_sample_completion_excludes_the_prompt_prefix():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.0)
    completion = sample_completion(model, tokenizer, "hello", config)
    assert "<|user|>" not in completion
    assert "<|assistant|>" not in completion


def test_generate_pairs_produces_two_completions_per_question():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.7)
    rows = generate_pairs(model, tokenizer, ["hello", "bye"], config)
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == {"prompt", "completion_a", "completion_b"}


def test_generate_pairs_writes_rows_incrementally_to_output_path(tmp_path):
    # Regression test: an earlier version held every row in memory and only wrote the
    # output file once the whole run finished, so a crash mid-run lost all completed
    # generation with no partial file to recover -- mirrors judge.py's already-fixed
    # incremental-output pattern.
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.7)
    output_path = tmp_path / "pairs_raw.jsonl"

    rows = generate_pairs(model, tokenizer, ["hello", "bye"], config, output_path=output_path)

    written = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert written == rows


def test_generate_pairs_resume_from_skips_already_generated_questions():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.7)

    rows = generate_pairs(model, tokenizer, ["hello", "bye", "hello"], config, resume_from=1)

    assert len(rows) == 2
    assert [row["prompt"] for row in rows] == ["bye", "hello"]


def test_generate_pairs_resume_from_appends_instead_of_overwriting_output(tmp_path):
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.7)
    output_path = tmp_path / "pairs_raw.jsonl"
    output_path.write_text(json.dumps({"prompt": "hello", "completion_a": "x", "completion_b": "y"}) + "\n")

    generate_pairs(
        model, tokenizer, ["hello", "bye"], config, output_path=output_path, resume_from=1
    )

    written = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(written) == 2
    assert written[0]["prompt"] == "hello"
    assert written[1]["prompt"] == "bye"


def test_generate_pairs_logs_progress(caplog):
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.7)

    with caplog.at_level(logging.INFO, logger="llmtrain.generate_pairs"):
        generate_pairs(model, tokenizer, ["hello", "bye"], config, progress_interval=1)

    assert any("1/2" in record.message for record in caplog.records)
    assert any("2/2" in record.message for record in caplog.records)
