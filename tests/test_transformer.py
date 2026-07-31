import pytest
import torch
from torch import nn

from llmtrain.model.cache import KVCache
from llmtrain.model.transformer import (
    MLP,
    CausalSelfAttention,
    TransformerLM,
    _rotary_cos_sin,
    apply_rotary,
)
from llmtrain.training.config import ModelConfig


def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, max_seq_len=6, dropout=0.0)


def test_forward_produces_correct_output_shape():
    model = TransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (3, 6))
    logits = model(input_ids)
    assert logits.shape == (3, 6, 16)


def test_backward_populates_gradients_for_every_parameter():
    model = TransformerLM(_tiny_config())
    input_ids = torch.randint(0, 16, (2, 6))
    logits = model(input_ids)
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_uses_rmsnorm_not_layernorm():
    model = TransformerLM(_tiny_config())
    assert isinstance(model.blocks[0].ln1, nn.RMSNorm)
    assert isinstance(model.blocks[0].ln2, nn.RMSNorm)
    assert isinstance(model.ln_f, nn.RMSNorm)


def test_head_weight_is_tied_to_token_embedding():
    model = TransformerLM(_tiny_config())
    assert model.head.weight is model.token_emb.weight


def test_tied_weight_receives_gradient():
    model = TransformerLM(_tiny_config())
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


def test_model_output_depends_on_token_order():
    torch.manual_seed(0)
    model = TransformerLM(_tiny_config()).eval()
    with torch.no_grad():
        a = model(torch.tensor([[3, 7]]))[:, -1]
        b = model(torch.tensor([[7, 3]]))[:, -1]
    assert not torch.allclose(a, b, atol=1e-5)


def test_rope_is_identity_at_position_zero():
    head_dim = 4
    cos, sin = _rotary_cos_sin(
        seq_len=1,
        head_dim=head_dim,
        theta=10000.0,
        position_offset=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x = torch.randn(1, 1, 1, head_dim)
    rotated = apply_rotary(x, cos, sin)
    assert torch.allclose(rotated, x, atol=1e-6)


def test_forward_handles_seq_len_larger_than_max_seq_len_used_at_construction():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_seq_len=6, dropout=0.0
    )
    model = TransformerLM(config)
    input_ids = torch.randint(0, 16, (2, 50))
    logits = model(input_ids)
    assert logits.shape == (2, 50, 16)


def test_gqa_projections_are_sized_for_fewer_kv_heads():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=6, dropout=0.0
    )
    attn = CausalSelfAttention(config)
    assert attn.q_proj.out_features == config.n_heads * attn.head_dim
    assert attn.kv_proj.out_features == 2 * config.n_kv_heads * attn.head_dim


def test_forward_works_with_grouped_query_attention():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=6, dropout=0.0
    )
    model = TransformerLM(config)
    input_ids = torch.randint(0, 16, (2, 6))
    logits = model(input_ids)
    assert logits.shape == (2, 6, 16)
    logits.sum().backward()


def test_n_heads_must_be_divisible_by_n_kv_heads():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=4, n_kv_heads=3, max_seq_len=6, dropout=0.0
    )
    with pytest.raises(ValueError):
        CausalSelfAttention(config)


def test_cached_decoding_matches_uncached_forward():
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=6, dropout=0.0
    )
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, 16, (1, 5))

    with torch.no_grad():
        full_logits = model(input_ids)

    cache = KVCache()
    cached_logits = []
    with torch.no_grad():
        for t in range(input_ids.shape[1]):
            step_logits = model(input_ids[:, t : t + 1], cache=cache)
            cached_logits.append(step_logits)
    cached_logits = torch.cat(cached_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5)


def test_multi_token_query_against_nonempty_cache_raises():
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=6, dropout=0.0
    )
    model = TransformerLM(config)
    model.eval()
    cache = KVCache()
    with torch.no_grad():
        model(torch.tensor([[1]]), cache=cache)
        with pytest.raises(ValueError):
            model(torch.tensor([[2, 3]]), cache=cache)


def test_multi_token_prefill_with_cache_matches_uncached():
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=16, d_model=8, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=6, dropout=0.0
    )
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, 16, (1, 5))

    # Full uncached forward
    with torch.no_grad():
        uncached_logits = model(input_ids)

    # Multi-token prefill into an initially-empty cache (simulates Task 9 prefill step)
    cache = KVCache()
    with torch.no_grad():
        prefill_logits = model(input_ids, cache=cache)

    # Prefill output should match uncached forward exactly
    assert torch.allclose(uncached_logits, prefill_logits, atol=1e-5)
