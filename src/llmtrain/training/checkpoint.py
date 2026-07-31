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
    # MinimalTransformerLM stores its ModelConfig as `self.config`; not every nn.Module
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


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict | None, dict | None]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["step"], checkpoint["dataset_state"], checkpoint.get("model_config")
