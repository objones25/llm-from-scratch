from dataclasses import asdict

import torch
from datasets import Dataset
from torch import nn

from llmtrain.model.transformer import TransformerLM
from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint
from llmtrain.training.config import ModelConfig


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
