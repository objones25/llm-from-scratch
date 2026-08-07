from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset_state: dict | None = None,
) -> None:
    # TransformerLM stores its ModelConfig as `self.config`; not every nn.Module
    # passed here has one (e.g. plain nn.Linear in tests), so this is best-effort.
    model_config = getattr(model, "config", None)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "dataset_state": dataset_state,
            "model_config": asdict(model_config) if model_config is not None else None,
        },
        path,
    )


def prune_old_checkpoints(checkpoint_dir: str | Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    checkpoints = sorted(
        Path(checkpoint_dir).glob("step_*.pt"),
        key=lambda p: int(p.stem.removeprefix("step_")),
    )
    for old_checkpoint in checkpoints[:-keep_last_n]:
        old_checkpoint.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict | None, dict | None]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    # Inference-only callers (generate.py) have no optimizer of their own, and
    # constructing a throwaway one just to satisfy this signature couples them to
    # train()'s optimizer shape (e.g. its param-group structure) for no reason.
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["step"], checkpoint["dataset_state"], checkpoint.get("model_config")
