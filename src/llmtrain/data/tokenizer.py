from collections.abc import Iterable

import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN]


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    # ByteLevel (GPT-2's scheme) marks word-initial bytes so decode() can losslessly
    # reconstruct spacing; Whitespace() pre-tokenization throws that boundary info away,
    # so no decoder can recover it and decode() falls back to joining every token with a
    # space. add_prefix_space=False keeps decode(encode(text)) exactly equal to text
    # (no leading space injected on the first word). Byte-level coverage also means
    # every possible input byte is representable, so [UNK] never actually fires.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def encode_batch(tokenizer: Tokenizer, texts: list[str], max_seq_len: int) -> torch.Tensor:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    tokenizer.enable_truncation(max_length=max_seq_len)
    tokenizer.enable_padding(pad_token=PAD_TOKEN, pad_id=pad_id, length=max_seq_len)
    encodings = tokenizer.encode_batch(texts)
    return torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
