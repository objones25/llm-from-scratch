import torch

from llmtrain.data.tokenizer import encode_batch, train_tokenizer


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
