import torch
from torch import nn

from llmtrain.model.transformer import MinimalTransformerLM
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
