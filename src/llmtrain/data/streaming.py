from collections.abc import Callable
from dataclasses import dataclass

from datasets import IterableDataset, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None
    split: str
    text_column: str = "text"


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tiny_shakespeare": DatasetSpec(
        path="Trelis/tiny-shakespeare", name=None, split="train", text_column="Text"
    ),
    "reformer_enwik8": DatasetSpec(path="reds0510/enwik8-processed", name=None, split="train"),
    "fineweb_edu": DatasetSpec(
        path="HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train"
    ),
}


def load_streaming_dataset(
    dataset_name: str,
    seed: int,
    buffer_size: int,
    load_fn: Callable[..., IterableDataset] = load_dataset,
) -> IterableDataset:
    spec = DATASET_REGISTRY[dataset_name]
    dataset = load_fn(spec.path, name=spec.name, split=spec.split, streaming=True)
    if spec.text_column != "text":
        dataset = dataset.rename_column(spec.text_column, "text")
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)
