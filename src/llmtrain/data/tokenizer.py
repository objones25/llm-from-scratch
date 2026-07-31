from collections.abc import Iterable

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN]


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def encode_batch(tokenizer: Tokenizer, texts: list[str], max_seq_len: int) -> torch.Tensor:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    tokenizer.enable_truncation(max_length=max_seq_len)
    tokenizer.enable_padding(pad_token=PAD_TOKEN, pad_id=pad_id, length=max_seq_len)
    encodings = tokenizer.encode_batch(texts)
    return torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
