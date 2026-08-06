import pytest
from datasets import Dataset

from llmtrain.data.streaming import DATASET_REGISTRY, DatasetSpec, load_streaming_dataset


def _fake_load_dataset(path, name, split, streaming):
    return Dataset.from_dict({"Text": [f"example {i}" for i in range(20)]}).to_iterable_dataset(
        num_shards=4
    )


def test_fineweb_edu_registry_entry_uses_sample_100bt_config():
    spec = DATASET_REGISTRY["fineweb_edu"]
    assert spec.path == "HuggingFaceFW/fineweb-edu"
    assert spec.name == "sample-100BT"
    assert spec.split == "train"


def test_load_streaming_dataset_shuffles_and_yields_every_example():
    dataset = load_streaming_dataset(
        "tiny_shakespeare", seed=42, buffer_size=5, load_fn=_fake_load_dataset
    )
    examples = list(dataset)
    assert len(examples) == 20
    assert all("text" in example for example in examples)


def test_dataset_spec_rejects_both_val_split_and_val_holdout_examples():
    with pytest.raises(ValueError):
        DatasetSpec(path="x", name=None, split="train", val_split="test", val_holdout_examples=5)


def test_dataset_spec_rejects_neither_val_split_nor_val_holdout_examples():
    with pytest.raises(ValueError):
        DatasetSpec(path="x", name=None, split="train")


def test_tiny_shakespeare_registry_entry_uses_native_val_split():
    spec = DATASET_REGISTRY["tiny_shakespeare"]
    assert spec.val_split == "test"
    assert spec.val_holdout_examples is None


def test_reformer_enwik8_and_fineweb_edu_registry_entries_carve_val_holdout():
    for name in ["reformer_enwik8", "fineweb_edu"]:
        spec = DATASET_REGISTRY[name]
        assert spec.val_split is None
        assert spec.val_holdout_examples == 1000
