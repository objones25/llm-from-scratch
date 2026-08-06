import argparse
import logging
import math
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from llmtrain.data.streaming import load_streaming_dataset
from llmtrain.data.tokenizer import PAD_TOKEN, encode_batch, train_tokenizer
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.checkpoint import load_checkpoint, save_checkpoint
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig

logger = logging.getLogger(__name__)


def select_device() -> torch.device:
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        ignore_index=pad_id,
    )


def make_collate_fn(tokenizer: Tokenizer, max_seq_len: int) -> Callable[[list[dict]], torch.Tensor]:
    def collate(examples: list[dict]) -> torch.Tensor:
        texts = [example["text"] for example in examples]
        return encode_batch(tokenizer, texts, max_seq_len)

    return collate


def get_lr(step: int, train_cfg: TrainConfig) -> float:
    # Linear warmup then cosine decay to min_lr (nanoGPT-style). Pure function of `step`,
    # so resuming from a checkpoint needs no separate scheduler state to persist.
    if step < train_cfg.warmup_steps:
        return train_cfg.lr * (step + 1) / train_cfg.warmup_steps
    if step >= train_cfg.max_steps:
        return train_cfg.min_lr
    decay_ratio = (step - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_cfg.min_lr + coeff * (train_cfg.lr - train_cfg.min_lr)


def train(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    resume_path: str | None = None,
) -> None:
    configure_logging(log_file=train_cfg.log_file)
    # Must be set before any CUDA allocation happens (the allocator reads it lazily on
    # first use) — reduces fragmentation-driven OOMs on long runs. No-op on MPS/CPU.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(train_cfg.seed)
    device = select_device()
    logger.info("training on device %s", device.type, extra={"device": device.type})
    if device.type == "cuda":
        # TF32 matmuls: near-free throughput on Ampere+ (A100) with negligible precision
        # loss for a model already training under bf16 autocast; no effect on MPS/CPU.
        torch.set_float32_matmul_precision("high")

    dataset = load_streaming_dataset(
        data_cfg.dataset_name, seed=train_cfg.seed, buffer_size=data_cfg.shuffle_buffer_size
    )
    sample_texts = [example["text"] for example in dataset.take(data_cfg.tokenizer_sample_size)]
    tokenizer = train_tokenizer(sample_texts, vocab_size=data_cfg.tokenizer_vocab_size)
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    model: torch.nn.Module = TransformerLM(model_cfg).to(device)
    if device.type == "cuda" and train_cfg.compile:
        # torch.compile's stub returns a broad callable type; the object is
        # still an nn.Module at runtime, hence the explicit annotation above.
        model = torch.compile(model)  # type: ignore[assignment]
    # GPT-3/LLaMA/nanoGPT-style two-group AdamW: exclude 1-D parameters (RMSNorm gains —
    # the only 1-D params left now that every nn.Linear is bias=False) from weight decay.
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    step = 0
    if resume_path is not None:
        # Training reconstructs the model from the current model_cfg, not the checkpoint's
        # persisted config — the returned model_config is only used by generate.py, which
        # rebuilds the model from scratch at inference time.
        step, dataset_state, _resumed_model_config = load_checkpoint(resume_path, model, optimizer)
        if dataset_state is not None:
            dataset.load_state_dict(dataset_state)
        logger.info("resumed from checkpoint at step %d", step, extra={"step": step})

    dataloader = DataLoader(
        dataset,  # type: ignore[arg-type]  # IterableDataset isn't in DataLoader's stub overloads, but is supported at runtime
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        # A ragged final batch would force torch.compile to recompile for the new shape,
        # spiking memory mid-run; dropping it keeps every batch's shape constant.
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len),
    )

    wandb.init(
        project=train_cfg.wandb_project,
        mode=train_cfg.wandb_mode,  # type: ignore[arg-type]
        config={**asdict(train_cfg), **asdict(model_cfg)},
    )

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Must save before the first encode_batch call: encode_batch mutates the tokenizer's
    # truncation/padding state via enable_truncation/enable_padding, and that mutated state
    # gets serialized into tokenizer.json. Saving later would silently persist the wrong
    # truncation/padding length for anything (e.g. generate.py) that loads this file.
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    logger.info("saved tokenizer to %s", checkpoint_dir / "tokenizer.json")

    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    assert pad_id is not None
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    micro_step = 0
    # A bare `for batch in dataloader` would stop as soon as the underlying stream is
    # exhausted, capping training at whatever step count one pass through the dataset
    # happens to reach — silently ignoring the rest of --max-steps. Wrapping in this
    # while re-iterates the dataset (a fresh epoch) whenever that happens. No-op for
    # datasets large enough to never exhaust within a normal run (reformer_enwik8,
    # fineweb_edu); this is what lets small datasets like tiny_shakespeare train for
    # more than one epoch.
    while step < train_cfg.max_steps:
        for batch in dataloader:
            input_ids = batch.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=train_cfg.use_amp
            ):
                logits = model(input_ids)
                loss = (
                    next_token_loss(logits, input_ids, pad_id)
                    / train_cfg.gradient_accumulation_steps
                )

            loss.backward()
            accumulated_loss += loss.item()
            micro_step += 1

            if micro_step % train_cfg.gradient_accumulation_steps != 0:
                continue

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            lr = get_lr(step, train_cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            wandb.log(
                {"loss": accumulated_loss, "lr": lr, "grad_norm": total_norm.item()}, step=step
            )
            logger.debug("step %d complete", step, extra={"step": step})
            accumulated_loss = 0.0

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
        default=DataConfig.dataset_name,
    )
    parser.add_argument("--shuffle-buffer-size", type=int, default=DataConfig.shuffle_buffer_size)
    parser.add_argument("--max-seq-len", type=int, default=DataConfig.max_seq_len)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=DataConfig.tokenizer_vocab_size)
    parser.add_argument(
        "--tokenizer-sample-size", type=int, default=DataConfig.tokenizer_sample_size
    )

    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--n-layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--n-heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--n-kv-heads", type=int, default=ModelConfig.n_kv_heads)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--rope-theta", type=float, default=ModelConfig.rope_theta)

    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TrainConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--beta1", type=float, default=TrainConfig.beta1)
    parser.add_argument("--beta2", type=float, default=TrainConfig.beta2)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--checkpoint-dir", type=str, default=TrainConfig.checkpoint_dir)
    parser.add_argument("--checkpoint-interval", type=int, default=TrainConfig.checkpoint_interval)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=TrainConfig.compile
    )
    parser.add_argument(
        "--use-amp", action=argparse.BooleanOptionalAction, default=TrainConfig.use_amp
    )
    parser.add_argument("--wandb-project", type=str, default=TrainConfig.wandb_project)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=TrainConfig.wandb_mode,
    )
    parser.add_argument("--log-file", type=str, default=TrainConfig.log_file)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    data_cfg = DataConfig(
        dataset_name=args.dataset,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_seq_len=args.max_seq_len,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        tokenizer_sample_size=args.tokenizer_sample_size,
    )
    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip=args.grad_clip,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        max_steps=args.max_steps,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        compile=args.compile,
        use_amp=args.use_amp,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        log_file=args.log_file,
    )
    train(data_cfg, model_cfg, train_cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
