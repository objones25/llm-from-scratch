import torch
from tokenizers import Tokenizer

IGNORE_INDEX = -100  # shared with training/train.py's loss functions


def format_turn(role: str, content: str) -> str:
    return f"<|{role}|>\n{content}\n"


def encode_chat_example(
    tokenizer: Tokenizer, messages: list[dict], pad_id: int, max_seq_len: int
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        turn_ids = tokenizer.encode(format_turn(message["role"], message["content"])).ids
        input_ids.extend(turn_ids)
        if message["role"] == "assistant":
            labels.extend(turn_ids)  # supervised
            input_ids.append(pad_id)  # stop signal
            labels.append(pad_id)  # ...and it's supervised too
        else:
            labels.extend([IGNORE_INDEX] * len(turn_ids))
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    pad_amount = max_seq_len - len(input_ids)
    input_ids.extend([pad_id] * pad_amount)
    labels.extend([IGNORE_INDEX] * pad_amount)  # tail filler: never supervised
    return input_ids, labels


def encode_chat_batch(
    tokenizer: Tokenizer, examples: list[dict], pad_id: int, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # "messages" is hardcoded here regardless of DatasetSpec.messages_column's value —
    # that field is only used elsewhere as an is-this-chat-data flag, not as a configurable
    # key name. Both current chat datasets (smoltalk, no_robots) use "messages"; a future
    # chat dataset with a differently-named column would need this function updated too.
    pairs = [
        encode_chat_example(tokenizer, ex["messages"], pad_id, max_seq_len) for ex in examples
    ]
    input_ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    labels = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return input_ids, labels
