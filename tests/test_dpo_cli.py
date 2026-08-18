import json

from llmtrain.training.dpo import build_dpo_config_from_args, load_dpo_dataset


def test_load_dpo_dataset_formats_prompt_with_chat_tags(tmp_path):
    pairs_path = tmp_path / "pairs_dpo.jsonl"
    pairs_path.write_text(
        json.dumps({"prompt": "what is 2+2", "chosen": "4", "rejected": "5"}) + "\n"
    )

    dataset = load_dpo_dataset(pairs_path)

    assert len(dataset) == 1
    row = dataset[0]
    assert row["prompt"] == "<|user|>\nwhat is 2+2\n<|assistant|>\n"
    assert row["chosen"] == "4"
    assert row["rejected"] == "5"


def test_load_dpo_dataset_reads_multiple_lines(tmp_path):
    pairs_path = tmp_path / "pairs_dpo.jsonl"
    rows = [
        {"prompt": "q1", "chosen": "a1", "rejected": "r1"},
        {"prompt": "q2", "chosen": "a2", "rejected": "r2"},
    ]
    pairs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    dataset = load_dpo_dataset(pairs_path)

    assert len(dataset) == 2


def test_build_dpo_config_from_args_maps_cli_flags_to_dpo_config_fields():
    import argparse

    args = argparse.Namespace(
        checkpoint_dir="/tmp/out",
        beta=0.2,
        loss_type="sigmoid",
        learning_rate=1e-6,
        num_train_epochs=2,
        max_length=512,
        batch_size=8,
    )

    dpo_config = build_dpo_config_from_args(args)

    assert dpo_config.output_dir == "/tmp/out"
    assert dpo_config.beta == 0.2
    assert dpo_config.loss_type == ["sigmoid"]
    assert dpo_config.learning_rate == 1e-6
    assert dpo_config.num_train_epochs == 2
    assert dpo_config.max_length == 512
    assert dpo_config.per_device_train_batch_size == 8
    assert dpo_config.gradient_checkpointing is False
    assert dpo_config.save_strategy == "no"
