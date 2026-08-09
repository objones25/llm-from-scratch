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
        DatasetSpec(path="x", name=None, split="train", text_column="Text", val_holdout_examples=5),
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
        return Dataset.from_dict({"text": [f"example {i}" for i in range(20)]}).to_iterable_dataset(
            num_shards=4
        )

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


def test_carve_path_val_is_materialized_and_does_not_disturb_train_state_dict(monkeypatch):
    # Regression test for a critical bug: shuffled.take(n)/shuffled.skip(n) both wrap the
    # *same* underlying _BaseExamplesIterable object (datasets doesn't copy it). If
    # val_dataset were left as a lazy IterableDataset, iterating it mid-training (as
    # evaluate() does) would disturb that shared object's internal state-tracking,
    # silently corrupting train_dataset.state_dict() from that point on — every
    # checkpoint saved after the first eval call would record a stale, rewound stream
    # position, so --resume would silently rewind and retrain on already-seen data.
    #
    # The fix materializes val_dataset into a plain list[dict] for both the carve path
    # and the native-split path. A plain list can never share mutable streaming state
    # with anything by construction, which eliminates this bug class entirely — so the
    # property this test locks in is simply "val_dataset is a list, not an
    # IterableDataset", for both paths. (A full repro that iterates val mid-consumption
    # of train and diffs state_dict() before/after is also included below for extra
    # confidence, verified to fail against the pre-fix lazy .take()/.skip() return.)
    monkeypatch.setitem(
        DATASET_REGISTRY,
        "shared_state_test",
        DatasetSpec(path="x", name=None, split="train", text_column="Text", val_holdout_examples=5),
    )
    train_dataset, val_dataset = load_streaming_datasets(
        "shared_state_test", seed=42, buffer_size=5, load_fn=_fake_load_dataset
    )
    assert isinstance(val_dataset, list)

    train_iterator = iter(train_dataset)
    consumed_before = [next(train_iterator) for _ in range(3)]
    state_before_eval = train_dataset.state_dict()

    # Simulate evaluate() fully iterating val mid-training, more than once (evaluate()
    # is called repeatedly over the course of training, re-iterating the same val list
    # every time).
    list(val_dataset)
    list(val_dataset)

    state_after_eval = train_dataset.state_dict()
    assert state_after_eval == state_before_eval

    remaining = [example["text"] for example in train_iterator]
    consumed_texts = [example["text"] for example in consumed_before]
    full_texts = [example["text"] for example in train_dataset.skip(0)]
    assert consumed_texts + remaining == full_texts


def test_native_split_val_is_also_materialized(monkeypatch):
    def _fake_load_dataset_by_split(path, name, split, streaming):
        texts = [f"{split}-example-{i}" for i in range(5)]
        return Dataset.from_dict({"text": texts}).to_iterable_dataset(num_shards=1)

    monkeypatch.setitem(
        DATASET_REGISTRY,
        "native_materialize_test",
        DatasetSpec(path="x", name=None, split="train", val_split="validation"),
    )
    _train_dataset, val_dataset = load_streaming_datasets(
        "native_materialize_test", seed=42, buffer_size=5, load_fn=_fake_load_dataset_by_split
    )
    assert isinstance(val_dataset, list)


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


def test_dataset_spec_defaults_messages_column_to_none():
    spec = DatasetSpec(path="x", name=None, split="train", val_split="test")
    assert spec.messages_column is None


def test_smoltalk_registry_entry_uses_messages_column_and_native_val_split():
    spec = DATASET_REGISTRY["smoltalk"]
    assert spec.path == "HuggingFaceTB/smoltalk"
    assert spec.name == "all"
    assert spec.messages_column == "messages"
    assert spec.val_split == "test"
    assert spec.val_holdout_examples is None


def test_no_robots_registry_entry_uses_messages_column_and_native_val_split():
    spec = DATASET_REGISTRY["no_robots"]
    assert spec.path == "HuggingFaceH4/no_robots"
    assert spec.name == "default"
    assert spec.messages_column == "messages"
    assert spec.val_split == "test"
    assert spec.val_holdout_examples is None


def test_load_streaming_datasets_skips_rename_when_messages_column_is_set(monkeypatch):
    def _fake_chat_load_dataset(path, name, split, streaming):
        return Dataset.from_dict(
            {"messages": [[{"role": "user", "content": f"hi {i}"}] for i in range(10)]}
        ).to_iterable_dataset(num_shards=1)

    monkeypatch.setitem(
        DATASET_REGISTRY,
        "chat_test",
        DatasetSpec(
            path="x",
            name=None,
            split="train",
            text_column="Text",  # would normally trigger rename_column("Text", "text")
            messages_column="messages",
            val_split="train",
        ),
    )
    train_dataset, _val_dataset = load_streaming_datasets(
        "chat_test", seed=42, buffer_size=5, load_fn=_fake_chat_load_dataset
    )

    examples = list(train_dataset)
    assert all("messages" in example for example in examples)
