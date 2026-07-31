from datasets import Dataset

from llmtrain.data.streaming import DATASET_REGISTRY, load_streaming_dataset


def _fake_load_dataset(path, name, split, streaming):
    return Dataset.from_dict(
        {"Text": [f"example {i}" for i in range(20)]}
    ).to_iterable_dataset(num_shards=4)


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
