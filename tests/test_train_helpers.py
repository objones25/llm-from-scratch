import torch

from llmtrain.data.tokenizer import train_tokenizer
from llmtrain.training.train import make_collate_fn, next_token_loss, select_device


def test_select_device_returns_a_torch_device():
    device = select_device()
    assert isinstance(device, torch.device)


def test_next_token_loss_is_near_zero_for_perfect_predictions():
    vocab_size = 4
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, vocab_size), -100.0)
    for position, target_id in enumerate(input_ids[0, 1:]):
        logits[0, position, target_id] = 100.0
    loss = next_token_loss(logits, input_ids, pad_id=99)
    assert loss.item() < 0.01


def test_make_collate_fn_encodes_a_batch_of_examples():
    texts = ["hello world", "hello there", "the quick brown fox"]
    tokenizer = train_tokenizer(texts, vocab_size=50)
    collate = make_collate_fn(tokenizer, max_seq_len=5)
    batch = collate([{"text": "hello world"}, {"text": "hello there"}])
    assert batch.shape == (2, 5)
    assert batch.dtype == torch.long
