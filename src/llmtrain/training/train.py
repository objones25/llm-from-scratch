from collections.abc import Callable

import torch
from torch.nn import functional as F

from llmtrain.data.tokenizer import encode_batch


def select_device() -> torch.device:
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
    )


def make_collate_fn(tokenizer, max_seq_len: int) -> Callable[[list[dict]], torch.Tensor]:
    def collate(examples: list[dict]) -> torch.Tensor:
        texts = [example["text"] for example in examples]
        return encode_batch(tokenizer, texts, max_seq_len)

    return collate
