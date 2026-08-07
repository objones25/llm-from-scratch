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
    "reformer_enwik8": DatasetSpec(
        path="reds0510/enwik8-processed", name=None, split="train", val_holdout_examples=1000
    ),
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
) -> tuple[IterableDataset, list[dict]]:
    spec = DATASET_REGISTRY[dataset_name]
    dataset = load_fn(spec.path, name=spec.name, split=spec.split, streaming=True)
    if spec.text_column != "text":
        dataset = dataset.rename_column(spec.text_column, "text")
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    if spec.val_split is not None:
        val_dataset = load_fn(spec.path, name=spec.name, split=spec.val_split, streaming=True)
        if spec.text_column != "text":
            val_dataset = val_dataset.rename_column(spec.text_column, "text")
        return shuffled, list(val_dataset)

    # Materialized (not left lazy via .take()) because .take()/.skip() on the same
    # `shuffled` object share its underlying _BaseExamplesIterable — iterating the val
    # split mid-training (inside evaluate()) disturbs that shared object's internal
    # state-tracking, silently corrupting train_dataset.state_dict() from that point on
    # (every checkpoint after the first eval call records a stale, rewound stream
    # position, so --resume silently retrains on already-seen data). A plain list can
    # never share mutable streaming state with anything, which eliminates the bug class
    # by construction. val_holdout_examples is always small (<=1000 by registry
    # default), so materializing is cheap, and it also avoids re-streaming the val set
    # from the network on every evaluate() call.
    # val_holdout_examples is guaranteed non-None here by __post_init__'s mutual-exclusivity
    # check (this branch is only reached when val_split is None), but that's a runtime
    # invariant type checkers can't see across the two fields.
    val_dataset = shuffled.take(spec.val_holdout_examples)  # type: ignore[arg-type]
    train_dataset = shuffled.skip(spec.val_holdout_examples)  # type: ignore[arg-type]
    return train_dataset, list(val_dataset)
