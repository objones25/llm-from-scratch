from dataclasses import asdict

import pytest
import torch
from datasets import Dataset
from torch import nn

import llmtrain.training.checkpoint as checkpoint_module
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.checkpoint import load_checkpoint, prune_old_checkpoints, save_checkpoint
from llmtrain.training.config import ModelConfig


def test_prune_old_checkpoints_keeps_only_the_most_recent_n(tmp_path):
    for step in [10, 20, 30, 40, 50]:
        (tmp_path / f"step_{step}.pt").write_bytes(b"")

    prune_old_checkpoints(tmp_path, keep_last_n=3)

    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_30.pt", "step_40.pt", "step_50.pt"]


def test_prune_old_checkpoints_disabled_when_keep_last_n_is_zero_or_negative(tmp_path):
    for step in [10, 20, 30]:
        (tmp_path / f"step_{step}.pt").write_bytes(b"")

    prune_old_checkpoints(tmp_path, keep_last_n=0)

    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_10.pt", "step_20.pt", "step_30.pt"]


def test_prune_old_checkpoints_is_noop_when_fewer_checkpoints_than_keep_last_n(tmp_path):
    for step in [10, 20]:
        (tmp_path / f"step_{step}.pt").write_bytes(b"")

    prune_old_checkpoints(tmp_path, keep_last_n=5)

    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_10.pt", "step_20.pt"]


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()
    original_weight = model.weight.detach().clone()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=7, dataset_state={"shard": 2})

    new_model = nn.Linear(4, 2)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    step, dataset_state, model_config = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert step == 7
    assert dataset_state == {"shard": 2}
    assert torch.equal(new_model.weight, original_weight)
    assert model_config is None


def test_load_checkpoint_without_optimizer_restores_model_only(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()
    original_weight = model.weight.detach().clone()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=7)

    new_model = nn.Linear(4, 2)
    step, _, _ = load_checkpoint(checkpoint_path, new_model)

    assert step == 7
    assert torch.equal(new_model.weight, original_weight)


def test_load_checkpoint_without_optimizer_tolerates_mismatched_param_group_shape(tmp_path):
    # The exact bug this guards against: generate.py has no reason to build an
    # optimizer whose param-group structure must match whatever train() used (e.g.
    # a two-group decay/no-decay AdamW) just to load a checkpoint for inference.
    model = nn.Linear(4, 2)
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=1)

    new_model = nn.Linear(4, 2)
    # No optimizer at all — not even a single-group one — and it must still work.
    load_checkpoint(checkpoint_path, new_model)


def test_checkpoint_round_trip_preserves_model_config(tmp_path):
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        dropout=0.0,
        rope_theta=5000.0,
    )
    model = TransformerLM(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=1)

    new_model = TransformerLM(config)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    _, _, loaded_model_config = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert loaded_model_config == asdict(config)


def test_save_checkpoint_retries_transient_write_failure_and_succeeds(tmp_path, monkeypatch):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    real_torch_save = torch.save
    calls = {"count": 0}

    def flaky_save(obj, path, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("basic_ios::clear: iostream error")
        return real_torch_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "save", flaky_save)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda seconds: None)

    checkpoint_path = tmp_path / "step_1.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=1)

    assert calls["count"] == 2
    assert checkpoint_path.exists()
    assert not checkpoint_path.with_name("step_1.pt.tmp").exists()


def test_save_checkpoint_raises_and_cleans_up_tmp_file_after_exhausting_retries(
    tmp_path, monkeypatch
):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def always_fails(obj, path, *args, **kwargs):
        raise RuntimeError("basic_ios::clear: iostream error")

    monkeypatch.setattr(checkpoint_module.torch, "save", always_fails)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda seconds: None)

    checkpoint_path = tmp_path / "step_1.pt"
    with pytest.raises(RuntimeError, match="iostream error"):
        save_checkpoint(checkpoint_path, model, optimizer, step=1)

    assert not checkpoint_path.exists()
    assert not checkpoint_path.with_name("step_1.pt.tmp").exists()


def test_checkpoint_preserves_iterable_dataset_resume_position(tmp_path):
    def make_dataset():
        return Dataset.from_dict({"value": list(range(10))}).to_iterable_dataset(num_shards=2)

    original = make_dataset()
    original_iter = iter(original)
    seen = [next(original_iter)["value"] for _ in range(4)]
    state = original.state_dict()

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=4, dataset_state=state)

    new_model = nn.Linear(1, 1)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    _, restored_state, _ = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    resumed = make_dataset()
    resumed.load_state_dict(restored_state)
    remaining = [example["value"] for example in resumed]

    assert set(seen).isdisjoint(remaining)
    assert len(seen) + len(remaining) == 10
