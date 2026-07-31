import torch
from torch import nn

from llmtrain.model.transformer import MLP, MinimalTransformerLM, _rotary_cos_sin, apply_rotary
from llmtrain.training.config import ModelConfig


def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, max_seq_len=6, dropout=0.0)


def test_forward_produces_correct_output_shape():
    model = MinimalTransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (3, 6))
    logits = model(input_ids)
    assert logits.shape == (3, 6, 16)


def test_backward_populates_gradients_for_every_parameter():
    model = MinimalTransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (2, 6))
    logits = model(input_ids)
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_uses_rmsnorm_not_layernorm():
    model = MinimalTransformerLM(_tiny_config())
    assert isinstance(model.blocks[0].ln1, nn.RMSNorm)
    assert isinstance(model.blocks[0].ln2, nn.RMSNorm)
    assert isinstance(model.ln_f, nn.RMSNorm)


def test_head_weight_is_tied_to_token_embedding():
    model = MinimalTransformerLM(_tiny_config())
    assert model.head.weight is model.token_emb.weight


def test_tied_weight_receives_gradient():
    model = MinimalTransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (2, 6))
    logits = model(input_ids)
    logits.sum().backward()
    assert model.token_emb.weight.grad is not None


def test_mlp_output_shape_matches_input():
    config = _tiny_config()
    mlp = MLP(config)
    x = torch.randn(3, 6, config.d_model)
    out = mlp(x)
    assert out.shape == x.shape


def test_mlp_uses_swiglu_hidden_dim_formula():
    config = _tiny_config()
    mlp = MLP(config)
    expected_d_ff = int(2 / 3 * 4 * config.d_model)
    assert mlp.w_gate.out_features == expected_d_ff
    assert mlp.w_up.out_features == expected_d_ff
    assert mlp.w_down.in_features == expected_d_ff


def test_rope_is_identity_at_position_zero():
    head_dim = 4
    cos, sin = _rotary_cos_sin(
        seq_len=1, head_dim=head_dim, theta=10000.0, position_offset=0,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    x = torch.randn(1, 1, 1, head_dim)
    rotated = apply_rotary(x, cos, sin)
    assert torch.allclose(rotated, x, atol=1e-6)


def test_forward_handles_seq_len_larger_than_max_seq_len_used_at_construction():
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_seq_len=6, dropout=0.0)
    model = MinimalTransformerLM(config)
    input_ids = torch.randint(0, 16, (2, 50))
    logits = model(input_ids)
    assert logits.shape == (2, 50, 16)
