from collections.abc import Callable
from dataclasses import dataclass

from datasets import IterableDataset, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    split: str
    text_column: str = "text"
    val_split: str | None = None
    val_holdout_examples: int | None = None

    def __post_init__(self) -> None:
        if (self.val_split is None) == (self.val_holdout_examples is None):
            raise ValueError("exactly one of val_split or val_holdout_examples must be set")


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tiny_shakespeare": DatasetSpec(
        path="Trelis/tiny-shakespeare",
        name=None,
        split="train",
        text_column="Text",
        val_split="test",
    ),
    "reformer_enwik8": DatasetSpec(path="reds0510/enwik8-processed", name=None, split="train", val_holdout_examples=1000),
    "fineweb_edu": DatasetSpec(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        val_holdout_examples=1000,
    ),
}


def load_streaming_datasets(
    dataset_name: str,
    seed: int,
    buffer_size: int,
    load_fn: Callable[..., IterableDataset] = load_dataset,
) -> tuple[IterableDataset, IterableDataset]:
    spec = DATASET_REGISTRY[dataset_name]
    dataset = load_fn(spec.path, name=spec.name, split=spec.split, streaming=True)
    if spec.text_column != "text":
        dataset = dataset.rename_column(spec.text_column, "text")
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    if spec.val_split is not None:
        val_dataset = load_fn(spec.path, name=spec.name, split=spec.val_split, streaming=True)
        if spec.text_column != "text":
            val_dataset = val_dataset.rename_column(spec.text_column, "text")
        return shuffled, val_dataset

    val_dataset = shuffled.take(spec.val_holdout_examples)
    train_dataset = shuffled.skip(spec.val_holdout_examples)
    return train_dataset, val_dataset
