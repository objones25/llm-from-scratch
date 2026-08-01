import pytest
import torch

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.training.config import TrainConfig
from llmtrain.training.train import get_lr, make_collate_fn, next_token_loss, select_device


def test_select_device_returns_a_torch_device():
    device = select_device()
    assert isinstance(device, torch.device)


def test_next_token_loss_is_near_zero_for_perfect_predictions():
    vocab_size = 4
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, vocab_size), -100.0)
    for position, target_id in enumerate(input_ids[0, 1:]):
        logits[0, position, target_id] = 100.0
    loss = next_token_loss(logits, input_ids, pad_id=99)
    assert loss.item() < 0.01


def test_make_collate_fn_encodes_a_batch_of_examples():
    texts = ["hello world", "hello there", "the quick brown fox"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    collate = make_collate_fn(tokenizer, max_seq_len=5)
    batch = collate([{"text": "hello world"}, {"text": "hello there"}])
    assert batch.shape == (2, 5)
    assert batch.dtype == torch.long


def test_get_lr_ramps_linearly_during_warmup():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(0, cfg) == pytest.approx(0.1)
    assert get_lr(9, cfg) == pytest.approx(1.0)


def test_get_lr_decays_smoothly_from_end_of_warmup():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(10, cfg) == pytest.approx(1.0)


def test_get_lr_decreases_monotonically_during_decay():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(50, cfg) > get_lr(90, cfg)


def test_get_lr_clamps_to_min_lr_after_max_steps():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(100, cfg) == pytest.approx(0.1)
    assert get_lr(150, cfg) == pytest.approx(0.1)


def test_get_lr_with_zero_warmup_starts_at_max_lr():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=0, max_steps=100)
    assert get_lr(0, cfg) == pytest.approx(1.0)
