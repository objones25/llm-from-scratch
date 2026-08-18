import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmtrain.data.tokenizer import PAD_TOKEN, train_tokenizer
from llmtrain.model.hf_wrapper import TransformerLMConfig, TransformerLMForCausalLM, wrap_tokenizer
from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint
from llmtrain.training.config import ModelConfig


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(vocab_size=16, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0)


def test_forward_returns_causal_lm_output_with_correct_logits_shape():
    hf_config = TransformerLMConfig.from_model_config(_tiny_model_config())
    model = TransformerLMForCausalLM(hf_config)
    input_ids = torch.randint(0, 16, (3, 6))

    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert isinstance(output, CausalLMOutputWithPast)
    assert output.logits.shape == (3, 6, 16)


def test_forward_ignores_attention_mask_and_still_produces_gradients():
    hf_config = TransformerLMConfig.from_model_config(_tiny_model_config())
    model = TransformerLMForCausalLM(hf_config)
    input_ids = torch.randint(0, 16, (2, 5))

    output = model(input_ids=input_ids, attention_mask=None)
    output.logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_wrap_tokenizer_preserves_encode_decode_round_trip_and_sets_special_tokens():
    tokenizer = train_tokenizer(["hello world", "hello there"], vocab_size=50)
    wrapped = wrap_tokenizer(tokenizer)

    assert wrapped("hello world")["input_ids"] == tokenizer.encode("hello world").ids
    assert wrapped.decode(tokenizer.encode("hello world").ids) == "hello world"
    assert wrapped.pad_token == PAD_TOKEN
    assert wrapped.eos_token == PAD_TOKEN
    assert wrapped.pad_token_id == tokenizer.token_to_id(PAD_TOKEN)


def test_wrapper_round_trips_through_the_existing_checkpoint_format(tmp_path):
    model_cfg = _tiny_model_config()
    hf_config = TransformerLMConfig.from_model_config(model_cfg)
    original = TransformerLMForCausalLM(hf_config)
    optimizer = torch.optim.AdamW(original.model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, original.model, optimizer, step=1)

    loaded = TransformerLMForCausalLM(hf_config)
    load_checkpoint(checkpoint_path, loaded.model)

    for p_orig, p_loaded in zip(original.model.parameters(), loaded.model.parameters()):
        assert torch.equal(p_orig, p_loaded)
