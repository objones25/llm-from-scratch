import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from trl import (
    DPOConfig,  # type: ignore
    DPOTrainer,  # type: ignore
)

from llmtrain.data.chat import format_prompt
from llmtrain.logging_config import configure_logging
from llmtrain.model.hf_wrapper import (
    TransformerLMConfig,
    TransformerLMForCausalLM,
    wrap_tokenizer,
)
from llmtrain.s3 import resolve_local_path, sibling_path
from llmtrain.training.checkpoint import (
    load_checkpoint,
    load_model_config_from_checkpoint,
    save_checkpoint,
)
from llmtrain.training.config import TrainConfig

logger = logging.getLogger(__name__)


def load_dpo_dataset(path: str | Path) -> Dataset:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    formatted = [
        {
            "prompt": format_prompt(row["prompt"]),
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in rows
    ]
    return Dataset.from_list(formatted)


def build_model_and_tokenizer(
    checkpoint_path: Path, tokenizer_path: Path
) -> tuple[
    TransformerLMForCausalLM,
    TransformerLMForCausalLM,
    PreTrainedTokenizerFast,
    Tokenizer,
]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_cfg = load_model_config_from_checkpoint(checkpoint_path, tokenizer.get_vocab_size())
    hf_config = TransformerLMConfig.from_model_config(model_cfg)
    model = TransformerLMForCausalLM(hf_config)
    load_checkpoint(checkpoint_path, model.model)

    # DPOTrainer's ref_model=None default fails for this wrapper (see
    # docs/superpowers/specs/2026-08-18-dpo-pipeline-design.md's "Reference model" section
    # and tests/test_hf_wrapper.py's regression test) -- always build one explicitly.
    ref_model = TransformerLMForCausalLM(hf_config)
    ref_model.load_state_dict(model.state_dict())

    wrapped_tokenizer = wrap_tokenizer(tokenizer)
    return model, ref_model, wrapped_tokenizer, tokenizer


def export_checkpoint(
    model: TransformerLMForCausalLM,
    tokenizer: Tokenizer,
    checkpoint_dir: Path,
    step: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # A throwaway optimizer -- save_checkpoint requires one, but generate.py (the only
    # consumer of this checkpoint) never loads optimizer state back (see checkpoint.py's
    # load_checkpoint: optimizer is optional on load, by design, for inference callers).
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
    save_checkpoint(checkpoint_dir / f"step_{step}.pt", model.model, optimizer, step=step)
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    logger.info("exported DPO checkpoint at step %d", step, extra={"step": step})


def build_dpo_config_from_args(args: argparse.Namespace) -> DPOConfig:
    return DPOConfig(
        output_dir=args.checkpoint_dir,
        beta=args.beta,
        loss_type=[args.loss_type],
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        # Our wrapper doesn't implement gradient checkpointing support (matches this
        # project's "deferred, not implemented" status for it elsewhere) -- DPOConfig's own
        # default (True) makes Trainer.train() call model.gradient_checkpointing_enable(),
        # which raises for any model that doesn't declare support for it.
        gradient_checkpointing=False,
        # A single short run (~2,000 pairs, 1 epoch) -- HF Trainer's own mid-run
        # checkpointing is unnecessary; export_checkpoint() saves the final result once,
        # through the existing checkpoint format, after trainer.train() completes.
        save_strategy="no",
        # W&B owns training metrics project-wide (CLAUDE.md's Logging & observability
        # section). report_to=["wandb"] enables HF Trainer's own WandbCallback, which
        # auto-calls wandb.init/log/finish during trainer.train() -- no manual
        # wandb.log() plumbing needed here, unlike train.py's hand-rolled loop, since
        # DPOTrainer owns the training loop itself. Project/mode are set via the
        # WANDB_PROJECT/WANDB_MODE env vars in main() (the mechanism HF's integration
        # reads), not DPOConfig fields.
        report_to=["wandb"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO-tune a checkpoint on judged preference pairs")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="SFT checkpoint to start from"
    )
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--pairs", type=str, required=True, help="path to pairs_dpo.jsonl")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="output directory")
    parser.add_argument("--beta", type=float, default=DPOConfig.beta)
    # DPOConfig.loss_type uses default_factory, not class-accessible.
    parser.add_argument("--loss-type", type=str, default="sigmoid")
    parser.add_argument("--learning-rate", type=float, default=DPOConfig.learning_rate)
    parser.add_argument("--num-train-epochs", type=int, default=int(DPOConfig.num_train_epochs))
    parser.add_argument("--max-length", type=int, default=DPOConfig.max_length)
    parser.add_argument("--batch-size", type=int, default=DPOConfig.per_device_train_batch_size)
    parser.add_argument("--log-file", type=str, default=TrainConfig.log_file)
    parser.add_argument("--wandb-project", type=str, default=TrainConfig.wandb_project)
    parser.add_argument("--wandb-mode", type=str, default=TrainConfig.wandb_mode)
    args = parser.parse_args()

    configure_logging(log_file=args.log_file)
    # HF Trainer's WandbCallback (enabled via report_to=["wandb"] in
    # build_dpo_config_from_args) reads project/mode from these env vars, not from
    # DPOConfig fields -- setdefault so an operator's own WANDB_PROJECT/WANDB_MODE (e.g.
    # set in .env) still takes precedence over these CLI defaults.
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    checkpoint_path = resolve_local_path(args.checkpoint)
    tokenizer_uri = args.tokenizer_path or sibling_path(args.checkpoint, "tokenizer.json")
    tokenizer_path = resolve_local_path(tokenizer_uri)

    model, ref_model, wrapped_tokenizer, tokenizer = build_model_and_tokenizer(
        checkpoint_path, tokenizer_path
    )
    dataset = load_dpo_dataset(args.pairs)
    dpo_config = build_dpo_config_from_args(args)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=wrapped_tokenizer,
    )
    trainer.train()

    export_checkpoint(model, tokenizer, Path(args.checkpoint_dir), step=trainer.state.global_step)


if __name__ == "__main__":
    main()
