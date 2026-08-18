import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from llmtrain.training.config import ModelConfig

logger = logging.getLogger(__name__)

_MAX_SAVE_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 5.0


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
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "dataset_state": dataset_state,
        "model_config": asdict(model_config) if model_config is not None else None,
    }

    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")

    # Checkpoints live on a network volume (see docs/training-guide.md's "dropped SSH
    # connection" note — confirmed in a real run that a transient write failure against
    # a network-mounted checkpoint dir raises RuntimeError: basic_ios::clear: iostream
    # error partway through torch.save, independent of anything on the local/SSH side).
    # Writing to a temp file and only renaming into place on success means a failed
    # attempt can never leave a corrupt step_N.pt behind; retrying absorbs transient
    # blips without losing the whole training process.
    for attempt in range(1, _MAX_SAVE_ATTEMPTS + 1):
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError):
            tmp_path.unlink(missing_ok=True)
            if attempt == _MAX_SAVE_ATTEMPTS:
                raise
            logger.warning(
                "checkpoint save to %s failed on attempt %d/%d, retrying",
                path,
                attempt,
                _MAX_SAVE_ATTEMPTS,
                exc_info=True,
            )
            time.sleep(_RETRY_DELAY_SECONDS)


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


def load_model_config_from_checkpoint(path: str | Path, vocab_size: int) -> ModelConfig:
    # Peek the checkpoint for its persisted model_config so architecture fields that
    # change numerics without changing tensor shapes (e.g. rope_theta) can't silently
    # drift from what the checkpoint was actually trained with. Older checkpoints saved
    # before model_config was persisted fall back to ModelConfig() defaults. Was
    # duplicated byte-for-byte across generate.py, generate_pairs.py, and
    # training/dpo.py -- consolidated here as the single source of truth.
    raw_checkpoint = torch.load(path, map_location="cpu")
    saved_model_config = raw_checkpoint.get("model_config")
    if saved_model_config is not None:
        return ModelConfig(**{**saved_model_config, "vocab_size": vocab_size})
    return ModelConfig(vocab_size=vocab_size)
