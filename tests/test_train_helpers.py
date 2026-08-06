import math

import pytest
import torch

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import ModelConfig, TrainConfig
from llmtrain.training.train import (
    evaluate,
    get_lr,
    make_collate_fn,
    next_token_loss,
    select_device,
)


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


def test_gradient_accumulation_matches_full_batch_gradient():
    torch.manual_seed(0)
    model_full = torch.nn.Linear(4, 2)
    model_accum = torch.nn.Linear(4, 2)
    model_accum.load_state_dict(model_full.state_dict())

    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    loss_full = torch.nn.functional.mse_loss(model_full(x), y)
    loss_full.backward()

    accumulation_steps = 4
    micro_batch_size = 2
    for i in range(accumulation_steps):
        start, end = i * micro_batch_size, (i + 1) * micro_batch_size
        micro_loss = (
            torch.nn.functional.mse_loss(model_accum(x[start:end]), y[start:end])
            / accumulation_steps
        )
        micro_loss.backward()

    for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
        assert torch.allclose(p_full.grad, p_accum.grad, atol=1e-5, rtol=1e-4)


def test_clip_grad_norm_caps_gradient_norm_and_returns_pre_clip_norm():
    model = torch.nn.Linear(4, 2)
    x = torch.randn(3, 4)
    loss = (model(x) * 1000.0).sum()
    loss.backward()

    manual_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()))
    max_norm = 1.0
    returned_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    assert returned_norm.item() == pytest.approx(manual_norm.item(), rel=1e-4)
    post_clip_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()))
    assert post_clip_norm.item() <= max_norm + 1e-4


def test_evaluate_returns_finite_float_and_restores_train_mode():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.train()

    # A plain list of batches stands in for a DataLoader here — evaluate() only ever
    # does `for batch in val_dataloader:`, so any iterable of pre-batched tensors works,
    # and this avoids pulling in real DataLoader/dataset machinery for a unit test.
    batch = torch.randint(0, 16, (2, 6))
    val_dataloader = [batch, batch]

    val_loss = evaluate(
        model,
        val_dataloader,
        pad_id=0,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
    )

    assert math.isfinite(val_loss)
    assert model.training is True


def test_evaluate_restores_eval_mode_if_model_was_already_in_eval_mode():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.eval()

    batch = torch.randint(0, 16, (2, 6))
    val_dataloader = [batch]

    evaluate(
        model,
        val_dataloader,
        pad_id=0,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
    )

    assert model.training is False
