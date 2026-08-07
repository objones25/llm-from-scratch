import pytest
from datasets import Dataset

from llmtrain.data.streaming import DATASET_REGISTRY, DatasetSpec, load_streaming_datasets


def _fake_load_dataset(path, name, split, streaming):
    return Dataset.from_dict({"Text": [f"example {i}" for i in range(20)]}).to_iterable_dataset(
        num_shards=4
    )


def test_fineweb_edu_registry_entry_uses_sample_100bt_config():
    spec = DATASET_REGISTRY["fineweb_edu"]
    assert spec.path == "HuggingFaceFW/fineweb-edu"
    assert spec.name == "sample-100BT"
    assert spec.split == "train"


def test_load_streaming_datasets_shuffles_and_yields_every_train_example():
    train_dataset, _val_dataset = load_streaming_datasets(
        "tiny_shakespeare", seed=42, buffer_size=5, load_fn=_fake_load_dataset
    )
    examples = list(train_dataset)
    assert len(examples) == 20
    assert all("text" in example for example in examples)


def test_load_streaming_datasets_carve_path_splits_train_and_val(monkeypatch):
    # text_column="Text" matches what _fake_load_dataset actually returns (capital T) —
    # without it, spec.text_column defaults to "text" and the rename_column step that
    # normally handles this mismatch gets skipped, so example["text"] would KeyError.
    monkeypatch.setitem(
        DATASET_REGISTRY,
        "carve_test",
        DatasetSpec(
            path="x", name=None, split="train", text_column="Text", val_holdout_examples=5
        ),
    )
    train_dataset, val_dataset = load_streaming_datasets(
        "carve_test", seed=42, buffer_size=5, load_fn=_fake_load_dataset
    )
    val_examples = list(val_dataset)
    train_examples = list(train_dataset)
    assert len(val_examples) == 5
    assert len(train_examples) == 15
    val_texts = {example["text"] for example in val_examples}
    train_texts = {example["text"] for example in train_examples}
    assert val_texts.isdisjoint(train_texts)


def test_load_streaming_datasets_native_split_path_uses_val_split_name(monkeypatch):
    def _fake_load_dataset_by_split(path, name, split, streaming):
        texts = [f"{split}-example-{i}" for i in range(5)]
        return Dataset.from_dict({"text": texts}).to_iterable_dataset(num_shards=1)

    monkeypatch.setitem(
        DATASET_REGISTRY,
        "native_test",
        DatasetSpec(path="x", name=None, split="train", val_split="validation"),
    )
    _train_dataset, val_dataset = load_streaming_datasets(
        "native_test", seed=42, buffer_size=5, load_fn=_fake_load_dataset_by_split
    )
    val_texts = {example["text"] for example in val_dataset}
    assert val_texts == {f"validation-example-{i}" for i in range(5)}


@pytest.mark.xfail(
    reason=(
        "datasets (v5.0.1) doesn't preserve shuffle-buffer contents across "
        "state_dict()/load_state_dict() — resuming drops up to buffer_size examples. "
        "Confirmed this is a property of .shuffle() itself (not .skip()): isolating "
        "skip() alone round-trips exactly, while shuffle() alone already loses "
        "buffer_size examples on resume, printing 'Loading a state dict of a shuffle "
        "buffer of a dataset without the buffer content. The shuffle buffer will be "
        "refilled before starting to yield new examples.' Not a bug in "
        "load_streaming_datasets; see CLAUDE.md's Dataset streaming & resume section."
    ),
    strict=True,
)
def test_shuffled_skip_dataset_resumes_correctly_via_state_dict():
    # This doesn't call load_streaming_datasets directly — it's a standalone check of
    # the exact mechanism the carve path depends on (shuffle().skip(n) + state_dict()/
    # load_state_dict()), since the `datasets` library's own docs don't explicitly
    # confirm this combination round-trips correctly for exact --resume.
    def _build_source():
        return Dataset.from_dict(
            {"text": [f"example {i}" for i in range(20)]}
        ).to_iterable_dataset(num_shards=4)

    n = 5
    dataset = _build_source().shuffle(seed=42, buffer_size=5).skip(n)

    consumed = []
    state = None
    for idx, example in enumerate(dataset):
        consumed.append(example)
        if idx == 2:
            state = dataset.state_dict()
            break

    resumed_dataset = _build_source().shuffle(seed=42, buffer_size=5).skip(n)
    resumed_dataset.load_state_dict(state)
    remaining = list(resumed_dataset)

    full = list(_build_source().shuffle(seed=42, buffer_size=5).skip(n))

    assert consumed + remaining == full


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
