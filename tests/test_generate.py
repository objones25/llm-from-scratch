import torch
from tokenizers import Tokenizer

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.generate import generate, generate_token_ids
from llmtrain.model.transformer import MinimalTransformerLM
from llmtrain.training.config import ModelConfig


def _tiny_setup() -> tuple[MinimalTransformerLM, Tokenizer]:
    tokenizer = train_tokenizer(
        ["hello world", "hello there", "world hello there"], vocab_size=32
    )
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=8,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        dropout=0.0,
    )
    model = MinimalTransformerLM(config)
    return model, tokenizer


def test_generate_token_ids_produces_requested_number_of_new_tokens():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    prompt_ids = tokenizer.encode("hello").ids
    output_ids = generate_token_ids(model, tokenizer, "hello", max_new_tokens=5, temperature=0.0)
    assert len(output_ids) == len(prompt_ids) + 5


def test_greedy_decoding_is_deterministic():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    output_a = generate(model, tokenizer, "hello", max_new_tokens=5, temperature=0.0)
    output_b = generate(model, tokenizer, "hello", max_new_tokens=5, temperature=0.0)
    assert output_a == output_b


def test_generate_token_ids_with_zero_max_new_tokens():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    prompt_ids = tokenizer.encode("hello").ids
    output_ids = generate_token_ids(model, tokenizer, "hello", max_new_tokens=0, temperature=0.0)
    assert len(output_ids) == len(prompt_ids)
    assert output_ids == prompt_ids
