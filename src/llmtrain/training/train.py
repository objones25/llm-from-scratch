import argparse
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader

from llmtrain.data.streaming import load_streaming_dataset
from llmtrain.data.tokenizer import encode_batch, train_tokenizer
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import MinimalTransformerLM
from llmtrain.training.checkpoint import save_checkpoint
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig

logger = logging.getLogger(__name__)


def select_device() -> torch.device:
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
    )


def make_collate_fn(tokenizer, max_seq_len: int) -> Callable[[list[dict]], torch.Tensor]:
    def collate(examples: list[dict]) -> torch.Tensor:
        texts = [example["text"] for example in examples]
        return encode_batch(tokenizer, texts, max_seq_len)

    return collate


def train(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    configure_logging(log_file=train_cfg.log_file)
    device = select_device()
    logger.info("training on device %s", device.type, extra={"device": device.type})

    dataset = load_streaming_dataset(
        data_cfg.dataset_name, seed=train_cfg.seed, buffer_size=data_cfg.shuffle_buffer_size
    )
    sample_texts = [example["text"] for example in dataset.take(200)]
    tokenizer = train_tokenizer(sample_texts, vocab_size=data_cfg.tokenizer_vocab_size)
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    model = MinimalTransformerLM(model_cfg).to(device)
    if device.type == "cuda" and train_cfg.compile:
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len),
    )

    wandb.init(
        project=train_cfg.wandb_project,
        mode=train_cfg.wandb_mode,  # type: ignore[arg-type]
        config=asdict(train_cfg),
    )

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    for batch in dataloader:
        input_ids = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=train_cfg.use_amp):
            logits = model(input_ids[:, :-1])
            loss = next_token_loss(logits, input_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()}, step=step)
        logger.debug("step %d complete", step, extra={"step": step})

        step += 1
        if step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{step}.pt",
                model,
                optimizer,
                step=step,
                dataset_state=dataset.state_dict(),
            )
            logger.info("saved checkpoint at step %d", step, extra={"step": step})
        if step >= train_cfg.max_steps:
            break

    wandb.finish()
    logger.info("training complete after %d steps", step, extra={"step": step})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the toy LLM")
    parser.add_argument(
        "--dataset",
        choices=["tiny_shakespeare", "reformer_enwik8", "fineweb_edu"],
        default="tiny_shakespeare",
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    data_cfg = DataConfig(dataset_name=args.dataset)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr)
    train(data_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
