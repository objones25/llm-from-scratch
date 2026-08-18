import torch
from tokenizers import Tokenizer

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.generate_pairs import format_prompt, generate_pairs, sample_completion
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import GenerationConfig, ModelConfig


def _tiny_setup() -> tuple[TransformerLM, Tokenizer]:
    tokenizer = train_tokenizer(
        ["<|user|>\nhello\n<|assistant|>\nhi there\n", "<|user|>\nbye\n<|assistant|>\nsee you\n"],
        vocab_size=64,
    )
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(), d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, dropout=0.0
    )
    model = TransformerLM(config)
    return model, tokenizer


def test_format_prompt_wraps_question_in_user_and_assistant_tags():
    assert format_prompt("hello") == "<|user|>\nhello\n<|assistant|>\n"


def test_sample_completion_excludes_the_prompt_prefix():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.0)
    completion = sample_completion(model, tokenizer, "hello", config)
    assert "<|user|>" not in completion
    assert "<|assistant|>" not in completion


def test_generate_pairs_produces_two_completions_per_question():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.7)
    rows = generate_pairs(model, tokenizer, ["hello", "bye"], config)
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == {"prompt", "completion_a", "completion_b"}
