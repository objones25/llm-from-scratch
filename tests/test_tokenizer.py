import torch

from llmtrain.data.tokenizer import PAD_TOKEN, encode_batch, train_tokenizer


def test_train_tokenizer_learns_a_vocab_from_texts():
    texts = ["hello world", "hello there", "the quick brown fox"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    assert tokenizer.get_vocab_size() > 0
    assert tokenizer.token_to_id("[PAD]") is not None


def test_encode_batch_returns_fixed_length_long_tensor():
    texts = ["hello world", "hello there", "the quick brown fox jumps"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    encoded = encode_batch(tokenizer, texts, max_seq_len=6)
    assert encoded.shape == (3, 6)
    assert encoded.dtype == torch.long


def test_decode_preserves_exact_text_and_spacing():
    texts = [
        "To be, or not to be, that is the question.",
        "Whether tis nobler in the mind to suffer",
    ]
    tokenizer = train_tokenizer(texts, vocab_size=300)
    for text in ["cousin nurse cover", "To be, or not to be", "hello world"]:
        ids = tokenizer.encode(text).ids
        assert tokenizer.decode(ids) == text


def test_encoding_never_produces_unk_for_untrained_characters():
    tokenizer = train_tokenizer(["hello world"], vocab_size=100)
    unk_id = tokenizer.token_to_id("[UNK]")
    ids = tokenizer.encode("日本語 emoji 🎉").ids
    assert unk_id not in ids


def test_pad_token_literal_text_round_trips_to_pad_id_when_appended():
    # TRL's DPOTrainer appends `tokenizer.eos_token` (literal text, not a raw id) to
    # chosen/rejected completions before encoding (see docs/superpowers/specs/
    # 2026-08-18-dpo-pipeline-design.md's "Dataset formatting" section). Setting
    # eos_token="[PAD]" on the wrapped tokenizer (model/hf_wrapper.py) only reinforces
    # the SFT-taught stop signal if this literal text round-trips to the single
    # dedicated pad token id, matching how encode_chat_example appends it as a raw id.
    tokenizer = train_tokenizer(["hello world", "hello there", "the quick brown fox"], vocab_size=50)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    plain_ids = tokenizer.encode("hello world").ids
    ids_with_pad_appended = tokenizer.encode("hello world" + PAD_TOKEN).ids

    assert ids_with_pad_appended[-1] == pad_id
    assert ids_with_pad_appended[:-1] == plain_ids
