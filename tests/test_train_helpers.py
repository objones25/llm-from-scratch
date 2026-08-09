import argparse
import math

import pytest
import torch
from datasets import Dataset

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig
from llmtrain.training.train import (
    build_configs_from_args,
    collect_micro_batches,
    compute_loss,
    evaluate,
    get_lr,
    load_or_train_tokenizer,
    make_collate_fn,
    next_token_loss,
    select_device,
    train_step,
)


def _fake_train_dataset(texts: list[str]):
    return Dataset.from_dict({"text": texts}).to_iterable_dataset(num_shards=1)


def test_select_device_returns_a_torch_device():
    device = select_device()
    assert isinstance(device, torch.device)


def test_next_token_loss_is_near_zero_for_perfect_predictions():
    vocab_size = 4
    labels = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, vocab_size), -100.0)
    for position, target_id in enumerate(labels[0, 1:]):
        logits[0, position, target_id] = 100.0
    loss = next_token_loss(logits, labels)
    assert loss.item() < 0.01


def test_next_token_loss_ignores_ignore_index_positions():
    vocab_size = 4
    logits = torch.randn(1, 3, vocab_size)
    labels_with_ignored = torch.tensor([[0, -100, 2]])
    labels_all_real = torch.tensor([[0, 1, 2]])
    # Wrong prediction at the ignored position shouldn't move the loss at all.
    loss_ignored = next_token_loss(logits, labels_with_ignored)
    loss_real = next_token_loss(logits, labels_all_real)
    assert loss_ignored.item() != pytest.approx(loss_real.item())


def test_make_collate_fn_encodes_a_batch_of_pretraining_examples():
    texts = ["hello world", "hello there", "the quick brown fox"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    collate = make_collate_fn(tokenizer, max_seq_len=5, messages_column=None)

    input_ids, labels = collate([{"text": "hello world"}, {"text": "hello there"}])

    assert input_ids.shape == (2, 5)
    assert input_ids.dtype == torch.long
    assert labels.shape == (2, 5)
    pad_id = tokenizer.token_to_id("[PAD]")
    assert torch.equal(labels == -100, input_ids == pad_id)


def test_make_collate_fn_encodes_a_batch_of_chat_examples():
    texts = ["<|user|>\nhi\n", "<|assistant|>\nhello\n"]
    tokenizer = train_tokenizer(texts, vocab_size=100)
    collate = make_collate_fn(tokenizer, max_seq_len=20, messages_column="messages")
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]

    input_ids, labels = collate(examples)

    assert input_ids.shape == (1, 20)
    assert labels.shape == (1, 20)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


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

    input_ids = torch.randint(0, 16, (2, 6))
    labels = input_ids.clone()
    val_dataloader = [(input_ids, labels), (input_ids, labels)]

    val_loss = evaluate(
        model,
        val_dataloader,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
        use_fused_ce=False,
    )

    assert math.isfinite(val_loss)
    assert model.training is True


def test_evaluate_restores_eval_mode_if_model_was_already_in_eval_mode():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.eval()

    input_ids = torch.randint(0, 16, (2, 6))
    labels = input_ids.clone()
    val_dataloader = [(input_ids, labels)]

    evaluate(
        model,
        val_dataloader,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_amp=False,
        use_fused_ce=False,
    )

    assert model.training is False


def test_compute_loss_non_fused_matches_direct_next_token_loss():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, 16, (2, 6))
    labels = input_ids.clone()

    with torch.no_grad():
        loss_via_compute_loss = compute_loss(model, input_ids, labels, use_fused_ce=False)
        logits = model(input_ids)
        loss_direct = next_token_loss(logits, labels)

    assert torch.allclose(loss_via_compute_loss, loss_direct, atol=1e-6)


def test_collect_micro_batches_collects_n_batches_in_order():
    dataloader = [torch.tensor([i]) for i in range(5)]
    data_iter = iter(dataloader)

    batches, data_iter = collect_micro_batches(dataloader, data_iter, 3)

    assert [b.item() for b in batches] == [0, 1, 2]


def test_collect_micro_batches_wraps_to_a_new_epoch_on_exhaustion():
    dataloader = [torch.tensor([i]) for i in range(2)]
    data_iter = iter(dataloader)

    batches, data_iter = collect_micro_batches(dataloader, data_iter, 3)

    assert [b.item() for b in batches] == [0, 1, 0]
    assert next(data_iter).item() == 1


def test_train_step_updates_parameters_and_zeros_grads_after():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)
    model = TransformerLM(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    train_cfg = TrainConfig(grad_clip=1.0, lr=1e-2, min_lr=1e-3, warmup_steps=0, max_steps=100)
    batch_1 = torch.randint(0, 16, (2, 6))
    batch_2 = torch.randint(0, 16, (2, 6))
    batches = [(batch_1, batch_1.clone()), (batch_2, batch_2.clone())]
    params_before = [p.clone() for p in model.parameters()]

    avg_loss, grad_norm, lr = train_step(
        model,
        optimizer,
        batches,
        train_cfg,
        device=torch.device("cpu"),
        autocast_dtype=None,
        use_fused_ce=False,
        step=0,
    )

    assert math.isfinite(avg_loss)
    assert grad_norm >= 0
    assert lr == pytest.approx(get_lr(0, train_cfg))
    assert any(
        not torch.equal(before, after) for before, after in zip(params_before, model.parameters())
    )
    assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters())


def test_build_configs_from_args_maps_every_field_to_its_dataclass():
    args = argparse.Namespace(
        dataset="tiny_shakespeare",
        shuffle_buffer_size=123,
        max_seq_len=64,
        tokenizer_vocab_size=500,
        tokenizer_sample_size=50,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        dropout=0.1,
        rope_theta=5000.0,
        batch_size=4,
        gradient_accumulation_steps=2,
        grad_clip=0.5,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=10,
        weight_decay=0.05,
        beta1=0.8,
        beta2=0.9,
        max_steps=100,
        seed=7,
        checkpoint_dir="ckpt",
        checkpoint_interval=10,
        keep_last_n_checkpoints=2,
        eval_interval=20,
        compile=False,
        use_amp=False,
        use_fused_ce=False,
        wandb_project="proj",
        wandb_mode="disabled",
        log_file="test.log",
        resume=None,
    )

    data_cfg, model_cfg, train_cfg = build_configs_from_args(args)

    assert data_cfg == DataConfig(
        dataset_name="tiny_shakespeare",
        shuffle_buffer_size=123,
        max_seq_len=64,
        tokenizer_vocab_size=500,
        tokenizer_sample_size=50,
    )
    assert model_cfg == ModelConfig(
        d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.1, rope_theta=5000.0
    )
    assert train_cfg == TrainConfig(
        batch_size=4,
        gradient_accumulation_steps=2,
        grad_clip=0.5,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=10,
        weight_decay=0.05,
        beta1=0.8,
        beta2=0.9,
        max_steps=100,
        seed=7,
        checkpoint_dir="ckpt",
        checkpoint_interval=10,
        keep_last_n_checkpoints=2,
        eval_interval=20,
        compile=False,
        use_amp=False,
        use_fused_ce=False,
        wandb_project="proj",
        wandb_mode="disabled",
        log_file="test.log",
    )


def test_load_or_train_tokenizer_trains_fresh_when_not_resuming():
    train_dataset = _fake_train_dataset(["hello world", "hello there", "the quick brown fox"])
    data_cfg = DataConfig(tokenizer_vocab_size=50, tokenizer_sample_size=3)

    tokenizer = load_or_train_tokenizer(None, train_dataset, data_cfg)

    assert tokenizer.get_vocab_size() > 0


def test_load_or_train_tokenizer_loads_saved_tokenizer_when_resuming(tmp_path):
    texts = ["hello world", "hello there"]
    train_dataset = _fake_train_dataset(texts)
    data_cfg = DataConfig(tokenizer_vocab_size=50, tokenizer_sample_size=2)
    saved_tokenizer = train_tokenizer(texts, vocab_size=50)
    saved_tokenizer.save(str(tmp_path / "tokenizer.json"))
    resume_path = str(tmp_path / "step_10.pt")

    loaded = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)

    assert loaded.get_vocab() == saved_tokenizer.get_vocab()


def test_load_or_train_tokenizer_falls_back_when_resuming_without_a_saved_tokenizer(tmp_path):
    texts = ["hello world", "hello there", "the quick brown fox"]
    train_dataset = _fake_train_dataset(texts)
    data_cfg = DataConfig(tokenizer_vocab_size=50, tokenizer_sample_size=3)
    resume_path = str(tmp_path / "step_10.pt")  # no tokenizer.json alongside it

    tokenizer = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)

    assert tokenizer.get_vocab_size() > 0
