import torch

from llmtrain.data.chat import IGNORE_INDEX, encode_chat_batch, encode_chat_example, format_turn
from llmtrain.data.tokenizer import PAD_TOKEN, train_tokenizer


def _tiny_tokenizer():
    texts = [
        format_turn("system", "be nice"),
        format_turn("user", "hi there"),
        format_turn("assistant", "hello"),
    ]
    return train_tokenizer(texts, vocab_size=200)


def test_format_turn_wraps_content_in_role_tags():
    assert format_turn("user", "hi") == "<|user|>\nhi\n"


def test_encode_chat_example_masks_non_assistant_turns_and_supervises_assistant_turns():
    tokenizer = _tiny_tokenizer()
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello"},
    ]
    max_seq_len = 64

    input_ids, labels = encode_chat_example(tokenizer, messages, pad_id, max_seq_len)

    assert len(input_ids) == max_seq_len
    assert len(labels) == max_seq_len

    system_ids = tokenizer.encode(format_turn("system", "be nice")).ids
    user_ids = tokenizer.encode(format_turn("user", "hi there")).ids
    assistant_ids = tokenizer.encode(format_turn("assistant", "hello")).ids

    non_assistant_len = len(system_ids) + len(user_ids)
    assert labels[:non_assistant_len] == [IGNORE_INDEX] * non_assistant_len
    assert input_ids[:non_assistant_len] == system_ids + user_ids

    assistant_start = non_assistant_len
    assistant_end = assistant_start + len(assistant_ids)
    assert labels[assistant_start:assistant_end] == assistant_ids
    assert input_ids[assistant_start:assistant_end] == assistant_ids

    # exactly one pad_id spliced in right after the assistant turn, and supervised
    assert input_ids[assistant_end] == pad_id
    assert labels[assistant_end] == pad_id

    # tail padding beyond that is never supervised
    assert all(label == IGNORE_INDEX for label in labels[assistant_end + 1 :])
    assert all(token_id == pad_id for token_id in input_ids[assistant_end + 1 :])


def test_encode_chat_example_truncates_to_exactly_max_seq_len():
    tokenizer = _tiny_tokenizer()
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    messages = [
        {"role": "user", "content": "hi there, how are you doing today my friend"},
        {"role": "assistant", "content": "I am doing great thanks for asking, how about you"},
    ]
    max_seq_len = 5

    input_ids, labels = encode_chat_example(tokenizer, messages, pad_id, max_seq_len)

    assert len(input_ids) == max_seq_len
    assert len(labels) == max_seq_len


def test_encode_chat_batch_stacks_examples_into_tensors():
    tokenizer = _tiny_tokenizer()
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    examples = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {
            "messages": [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "hi"},
            ]
        },
    ]

    input_ids, labels = encode_chat_batch(tokenizer, examples, pad_id, max_seq_len=16)

    assert input_ids.shape == (2, 16)
    assert labels.shape == (2, 16)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
