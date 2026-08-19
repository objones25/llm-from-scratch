import argparse
import logging
import math
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import torch
from datasets import IterableDataset
from tokenizers import Tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
from llmtrain.data.chat import IGNORE_INDEX, encode_chat_batch
from llmtrain.data.streaming import DATASET_REGISTRY, load_streaming_datasets
from llmtrain.data.tokenizer import PAD_TOKEN, encode_batch, train_tokenizer
from llmtrain.logging_config import configure_logging
from llmtrain.model.transformer import TransformerLM
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import load_checkpoint, prune_old_checkpoints, save_checkpoint
from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig

logger = logging.getLogger(__name__)

_Batch = TypeVar("_Batch")


def select_device() -> torch.device:
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def next_token_loss_fused(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    from liger_kernel.transformers import (  # type: ignore[import-not-found]
        LigerFusedLinearCrossEntropyLoss,
    )

    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return loss_fn(head_weight, shift_hidden, shift_labels)


def compute_loss(
    model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, use_fused_ce: bool
) -> torch.Tensor:
    if use_fused_ce:
        hidden = model(input_ids, return_hidden=True)  # type: ignore[misc]
        head_weight = model.token_emb.weight  # type: ignore[union-attr,attr-defined]
        return next_token_loss_fused(hidden, head_weight, labels)  # type: ignore[arg-type]
    logits = model(input_ids)
    return next_token_loss(logits, labels)


def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    use_amp: bool,
    use_fused_ce: bool,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for input_ids, labels in val_dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                losses.append(compute_loss(model, input_ids, labels, use_fused_ce).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def make_collate_fn(
    tokenizer: Tokenizer, max_seq_len: int, messages_column: str | None
) -> Callable[[list[dict]], tuple[torch.Tensor, torch.Tensor]]:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    assert pad_id is not None

    def collate(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        if messages_column is not None:
            return encode_chat_batch(tokenizer, examples, pad_id, max_seq_len)
        input_ids = encode_batch(tokenizer, [ex["text"] for ex in examples], max_seq_len)
        labels = input_ids.clone()
        labels[input_ids == pad_id] = IGNORE_INDEX
        return input_ids, labels

    return collate


def load_or_train_tokenizer(
    resume_path: str | None, train_dataset: IterableDataset, data_cfg: DataConfig
) -> Tokenizer:
    if resume_path is not None:
        tokenizer_path = resolve_local_path(sibling_path(resume_path, "tokenizer.json"))
        if tokenizer_path.exists():
            logger.info("loaded tokenizer from %s", tokenizer_path)
            return Tokenizer.from_file(str(tokenizer_path))
        logger.warning(
            "no tokenizer.json next to checkpoint %s; regenerating from the dataset "
            "stream instead of loading it",
            resume_path,
        )
    sample_texts = [
        example["text"] for example in train_dataset.take(data_cfg.tokenizer_sample_size)
    ]
    return train_tokenizer(sample_texts, vocab_size=data_cfg.tokenizer_vocab_size)


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


def collect_micro_batches(  # noqa: UP047 -- TypeVar kept per design (Task 3 brief), not PEP 695
    dataloader: Iterable[_Batch], data_iter: Iterator[_Batch], n: int
) -> tuple[list[_Batch], Iterator[_Batch]]:
    # A dataloader smaller than n (e.g. tiny_shakespeare with a large
    # --gradient-accumulation-steps) can wrap around more than once per call — each
    # StopIteration starts a fresh epoch rather than raising past a bare `next()`.
    batches: list[_Batch] = []
    for _ in range(n):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batches.append(batch)
    return batches, data_iter


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_cfg: TrainConfig,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    use_fused_ce: bool,
    step: int,
) -> tuple[float, float, float]:
    accumulated_loss = 0.0
    for input_ids, labels in batches:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=train_cfg.use_amp):
            loss = compute_loss(model, input_ids, labels, use_fused_ce) / len(batches)
        loss.backward()
        accumulated_loss += loss.item()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

    lr = get_lr(step, train_cfg)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    optimizer.zero_grad()

    return accumulated_loss, total_norm.item(), lr


def find_model_config_overrides(
    model_cfg: ModelConfig, saved_model_config: dict
) -> dict[str, tuple[object, object]]:
    return {
        field: (getattr(model_cfg, field), saved_model_config[field])
        for field in saved_model_config
        if field != "vocab_size" and getattr(model_cfg, field) != saved_model_config[field]
    }


def train(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    resume_path: str | None = None,
    init_from_checkpoint: str | None = None,
    tokenizer_path: str | None = None,
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

    train_dataset, val_dataset = load_streaming_datasets(
        data_cfg.dataset_name,
        seed=train_cfg.seed,
        buffer_size=data_cfg.shuffle_buffer_size,
    )
    init_checkpoint_path: Path | None = None
    if init_from_checkpoint is not None:
        # SFT always starts from pretrained weights with a fresh tokenizer loaded from
        # disk, never a freshly retrained one over smoltalk/no_robots text — the SFT run
        # must use the exact tokenizer the pretrained embeddings were trained with.
        init_checkpoint_path = resolve_local_path(init_from_checkpoint)
        tokenizer_uri = tokenizer_path or sibling_path(init_from_checkpoint, "tokenizer.json")
        tokenizer = Tokenizer.from_file(str(resolve_local_path(tokenizer_uri)))
        raw_checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
        saved_model_config = raw_checkpoint.get("model_config")
        if saved_model_config is not None:
            overrides = find_model_config_overrides(model_cfg, saved_model_config)
            if overrides:
                logger.warning(
                    "effective model architecture differs from checkpoint %s's; the "
                    "checkpoint's values win: %s",
                    init_from_checkpoint,
                    overrides,
                )
            model_cfg = ModelConfig(
                **{**saved_model_config, "vocab_size": tokenizer.get_vocab_size()}
            )
        else:
            model_cfg.vocab_size = tokenizer.get_vocab_size()
        del raw_checkpoint
    else:
        tokenizer = load_or_train_tokenizer(resume_path, train_dataset, data_cfg)
        model_cfg.vocab_size = tokenizer.get_vocab_size()

    model: torch.nn.Module = TransformerLM(model_cfg).to(device)
    if device.type == "cuda" and train_cfg.compile:
        # model.compile() (in-place) is the current PyTorch guidance over the older
        # functional torch.compile(model) wrapping — no reassignment needed, and
        # unlike the functional form it never wraps the model in an OptimizedModule,
        # so no _orig_mod.-prefixed state_dict keys, no attribute-proxying edge
        # cases, and generate.py (which always loads into a fresh, uncompiled model)
        # can never be affected by whatever wrapping strategy training used.
        model.compile()
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
    if init_from_checkpoint is not None:
        assert init_checkpoint_path is not None
        load_checkpoint(init_checkpoint_path, model, optimizer=None)
        logger.info("initialized weights from checkpoint %s", init_from_checkpoint)
    elif resume_path is not None:
        # Training reconstructs the model from the current model_cfg, not the checkpoint's
        # persisted config — the returned model_config is only used by generate.py, which
        # rebuilds the model from scratch at inference time.
        resume_checkpoint_path = resolve_local_path(resume_path)
        step, dataset_state, _resumed_model_config = load_checkpoint(
            resume_checkpoint_path, model, optimizer
        )
        if dataset_state is not None:
            train_dataset.load_state_dict(dataset_state)
        logger.info("resumed from checkpoint at step %d", step, extra={"step": step})

    messages_column = DATASET_REGISTRY[data_cfg.dataset_name].messages_column
    dataloader = DataLoader(
        train_dataset,  # type: ignore[arg-type]  # IterableDataset isn't in DataLoader's stub overloads, but is supported at runtime
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        # A ragged final batch would force torch.compile to recompile for the new shape,
        # spiking memory mid-run; dropping it keeps every batch's shape constant.
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len, messages_column),
    )
    # val_dataset is a plain list[dict] (materialized by load_streaming_datasets to avoid
    # sharing mutable streaming state with train_dataset) — a valid map-style dataset at
    # runtime (any Sequence with __getitem__/__len__ works), but list isn't a subtype of
    # the Dataset[T] the stub declares, hence the ignore. The DataLoader itself yields
    # (input_ids, labels) tuples once collate_fn runs, not the raw dict rows.
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset,  # type: ignore[arg-type]  # list[dict] is a valid map-style dataset at runtime but isn't Dataset[T]
        batch_size=train_cfg.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, data_cfg.max_seq_len, messages_column),
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

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    use_fused_ce_effective = train_cfg.use_fused_ce and device.type == "cuda"

    model.train()
    optimizer.zero_grad()
    # A bare `for batch in dataloader` would stop as soon as the underlying stream is
    # exhausted, capping training at whatever step count one pass through the dataset
    # happens to reach — silently ignoring the rest of --max-steps. collect_micro_batches
    # re-iterates the dataset (a fresh epoch) whenever that happens. No-op for datasets
    # large enough to never exhaust within a normal run (reformer_enwik8, fineweb_edu);
    # this is what lets small datasets like tiny_shakespeare train for more than one epoch.
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]] = iter(dataloader)
    while step < train_cfg.max_steps:
        batches, data_iter = collect_micro_batches(
            dataloader, data_iter, train_cfg.gradient_accumulation_steps
        )
        avg_loss, grad_norm, lr = train_step(
            model,
            optimizer,
            batches,
            train_cfg,
            device,
            autocast_dtype,
            use_fused_ce_effective,
            step,
        )

        wandb.log({"loss": avg_loss, "lr": lr, "grad_norm": grad_norm}, step=step)
        logger.debug("step %d complete", step, extra={"step": step})

        step += 1
        if step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{step}.pt",
                model,
                optimizer,
                step=step,
                dataset_state=train_dataset.state_dict(),
            )
            prune_old_checkpoints(checkpoint_dir, train_cfg.keep_last_n_checkpoints)
            logger.info("saved checkpoint at step %d", step, extra={"step": step})
        if step % train_cfg.eval_interval == 0:
            val_loss = evaluate(
                model,
                val_dataloader,
                device,
                autocast_dtype,
                train_cfg.use_amp,
                use_fused_ce_effective,
            )
            wandb.log({"val_loss": val_loss}, step=step)
            logger.info(
                "val_loss %.4f at step %d",
                val_loss,
                step,
                extra={"step": step, "val_loss": val_loss},
            )

    wandb.finish()
    logger.info("training complete after %d steps", step, extra={"step": step})


def build_configs_from_args(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, TrainConfig]:
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
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        eval_interval=args.eval_interval,
        compile=args.compile,
        use_amp=args.use_amp,
        use_fused_ce=args.use_fused_ce,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        log_file=args.log_file,
    )
    return data_cfg, model_cfg, train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the toy LLM")
    parser.add_argument(
        "--dataset",
        choices=["tiny_shakespeare", "reformer_enwik8", "fineweb_edu", "smoltalk", "no_robots"],
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
        "--keep-last-n-checkpoints", type=int, default=TrainConfig.keep_last_n_checkpoints
    )
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=TrainConfig.compile
    )
    parser.add_argument(
        "--use-amp", action=argparse.BooleanOptionalAction, default=TrainConfig.use_amp
    )
    parser.add_argument(
        "--use-fused-ce",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.use_fused_ce,
    )
    parser.add_argument("--wandb-project", type=str, default=TrainConfig.wandb_project)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=TrainConfig.wandb_mode,
    )
    parser.add_argument("--log-file", type=str, default=TrainConfig.log_file)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=str, default=None)
    resume_group.add_argument("--init-from-checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    args = parser.parse_args()

    if args.tokenizer_path and not args.init_from_checkpoint:
        parser.error("--tokenizer-path requires --init-from-checkpoint")
    if (
        DATASET_REGISTRY[args.dataset].messages_column is not None
        and not args.init_from_checkpoint
        and not args.resume
    ):
        parser.error(
            f"--dataset {args.dataset} is a chat dataset; chat datasets require "
            "--init-from-checkpoint (SFT always starts from a pretrained checkpoint) "
            "or --resume"
        )

    data_cfg, model_cfg, train_cfg = build_configs_from_args(args)
    train(
        data_cfg,
        model_cfg,
        train_cfg,
        resume_path=args.resume,
        init_from_checkpoint=args.init_from_checkpoint,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
