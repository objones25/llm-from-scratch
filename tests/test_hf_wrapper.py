import pytest
import torch
from datasets import Dataset
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import DPOConfig, DPOTrainer

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


def _tiny_dpo_setup():
    texts = [
        "hello world",
        "hello there friend",
        "the quick brown fox jumps over",
        "goodbye my old friend",
    ]
    tokenizer = train_tokenizer(texts, vocab_size=64)
    wrapped_tokenizer = wrap_tokenizer(tokenizer)
    hf_config = TransformerLMConfig(
        vocab_size=wrapped_tokenizer.vocab_size, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1
    )
    model = TransformerLMForCausalLM(hf_config)
    ref_model = TransformerLMForCausalLM(hf_config)
    ref_model.load_state_dict(model.state_dict())
    dataset = Dataset.from_dict(
        {
            "prompt": ["hello ", "hello ", "hello ", "hello "],
            "chosen": ["world", "there friend, the quick brown fox jumps over", "world", "there friend"],
            "rejected": ["there", "world", "there", "world"],
        }
    )
    return model, ref_model, wrapped_tokenizer, dataset


def test_dpo_trainer_batches_are_right_padded(tmp_path):
    model, ref_model, wrapped_tokenizer, dataset = _tiny_dpo_setup()
    dpo_config = DPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        report_to=[],
        use_cpu=True,
        max_length=64,
        gradient_checkpointing=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=wrapped_tokenizer,
    )

    batch = next(iter(trainer.get_train_dataloader()))

    # Right-padding: once a row's attention_mask hits 0 (padded), every later position in
    # that row must also be 0 -- the mask never goes back to 1 after the first pad.
    for row in batch["attention_mask"]:
        seen_pad = False
        for value in row.tolist():
            if value == 0:
                seen_pad = True
            elif seen_pad:
                pytest.fail("attention_mask has a real token after a padded position")


def test_dpo_trainer_requires_an_explicit_ref_model_for_this_wrapper(tmp_path):
    # Regression test for a real, confirmed bug: DPOTrainer's ref_model=None default tries
    # to reload a fresh reference model from a Hub repo id derived from the policy model's
    # config (create_model_from_path). Our wrapper is constructed directly in Python with
    # no Hub repo id, so that path raises -- which is exactly why training/dpo.py always
    # builds and passes a second TransformerLMForCausalLM instance explicitly.
    model, _ref_model, wrapped_tokenizer, dataset = _tiny_dpo_setup()
    dpo_config = DPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        report_to=[],
        use_cpu=True,
        max_length=64,
        gradient_checkpointing=False,
    )

    with pytest.raises(Exception):
        DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=wrapped_tokenizer,
        )
