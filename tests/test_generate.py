import pytest
import torch
from tokenizers import Tokenizer

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.generate import (
    _apply_repetition_penalty,
    _apply_top_k,
    _apply_top_p,
    _sample,
    generate,
    generate_token_ids,
)
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint
from llmtrain.training.config import GenerationConfig, ModelConfig


def _tiny_setup() -> tuple[TransformerLM, Tokenizer]:
    tokenizer = train_tokenizer(["hello world", "hello there", "world hello there"], vocab_size=32)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=8,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
    )
    model = TransformerLM(config)
    return model, tokenizer


def test_generate_token_ids_produces_requested_number_of_new_tokens():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    prompt_ids = tokenizer.encode("hello").ids
    output_ids = generate_token_ids(
        model, tokenizer, "hello", GenerationConfig(max_new_tokens=5, temperature=0.0)
    )
    assert len(output_ids) == len(prompt_ids) + 5


def test_greedy_decoding_is_deterministic():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    config = GenerationConfig(max_new_tokens=5, temperature=0.0)
    output_a = generate(model, tokenizer, "hello", config)
    output_b = generate(model, tokenizer, "hello", config)
    assert output_a == output_b


def test_generate_token_ids_with_zero_max_new_tokens():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    prompt_ids = tokenizer.encode("hello").ids
    output_ids = generate_token_ids(
        model, tokenizer, "hello", GenerationConfig(max_new_tokens=0, temperature=0.0)
    )
    assert len(output_ids) == len(prompt_ids)
    assert output_ids == prompt_ids


def test_generate_token_ids_raises_on_empty_prompt():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    with pytest.raises(ValueError):
        generate_token_ids(
            model, tokenizer, "", GenerationConfig(max_new_tokens=5, temperature=0.0)
        )


def test_generate_token_ids_restores_callers_training_mode():
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    model.train()
    generate_token_ids(
        model, tokenizer, "hello", GenerationConfig(max_new_tokens=3, temperature=0.0)
    )
    assert model.training is True


def test_repetition_penalty_divides_positive_seen_logit():
    logits = torch.tensor([[5.0, 1.0, -2.0]])
    penalized = _apply_repetition_penalty(logits, generated_ids=[0], penalty=2.0)
    assert penalized[0, 0].item() == pytest.approx(2.5)
    assert penalized[0, 1].item() == pytest.approx(1.0)
    assert penalized[0, 2].item() == pytest.approx(-2.0)


def test_repetition_penalty_multiplies_negative_seen_logit():
    logits = torch.tensor([[-2.0, 1.0]])
    penalized = _apply_repetition_penalty(logits, generated_ids=[0], penalty=2.0)
    assert penalized[0, 0].item() == pytest.approx(-4.0)


def test_repetition_penalty_is_noop_at_default_value():
    logits = torch.tensor([[5.0, 1.0]])
    penalized = _apply_repetition_penalty(logits, generated_ids=[0, 1], penalty=1.0)
    assert torch.equal(penalized, logits)


def test_top_k_keeps_only_top_k_tokens():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    filtered = _apply_top_k(logits, top_k=2)
    kept = filtered[0] > float("-inf")
    assert kept.tolist() == [True, True, False, False, False]


def test_top_k_is_noop_when_disabled():
    logits = torch.tensor([[5.0, 4.0, 3.0]])
    filtered = _apply_top_k(logits, top_k=0)
    assert torch.equal(filtered, logits)


def test_top_p_keeps_smallest_set_covering_cumulative_probability():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    filtered = _apply_top_p(logits, top_p=0.4)
    kept = filtered[0] > float("-inf")
    assert kept.tolist() == [True, False, False, False]


def test_top_p_is_noop_at_default_value():
    logits = torch.tensor([[5.0, 4.0, 3.0]])
    filtered = _apply_top_p(logits, top_p=1.0)
    assert torch.equal(filtered, logits)


def test_repetition_penalty_avoids_repeating_top_greedy_token():
    logits = torch.tensor([[5.0, 4.9, 1.0]])
    next_id = _sample(
        logits, generated_ids=[0], temperature=0.0, repetition_penalty=1.5, top_k=0, top_p=1.0
    )
    assert next_id == 1


def test_generate_works_after_checkpoint_and_tokenizer_round_trip(tmp_path):
    torch.manual_seed(0)
    model, tokenizer = _tiny_setup()
    # Matches train()'s real optimizer shape: a two-param-group AdamW (decay/no-decay
    # split for weight decay), not a single-group optimizer. generate.py has no reason
    # to construct an optimizer at all for inference — this test exists specifically to
    # catch the class of bug where load_checkpoint() required one anyway and its shape
    # silently had to match training's, which a single-group optimizer never would.
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )

    checkpoint_path = tmp_path / "step_1.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    save_checkpoint(checkpoint_path, model, optimizer, step=1)
    tokenizer.save(str(tokenizer_path))

    loaded_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    loaded_config = ModelConfig(
        vocab_size=loaded_tokenizer.get_vocab_size(),
        d_model=8,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
    )
    loaded_model = TransformerLM(loaded_config)
    load_checkpoint(checkpoint_path, loaded_model)

    output = generate(
        loaded_model, loaded_tokenizer, "hello", GenerationConfig(max_new_tokens=3, temperature=0.0)
    )
    assert isinstance(output, str)
