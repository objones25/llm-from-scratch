# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A toy LLM built from scratch and trained on `HuggingFaceFW/fineweb-edu`, using Hugging Face `tokenizers` and PyTorch. Full-scale training runs on a rented RunPod A100 GPU. The project is currently unscaffolded — this file is a spec for whoever builds it next, not a description of existing code.

A Hugging Face token is required for dataset/model access. Never commit it or write it into code — load it from the environment (e.g. `HF_TOKEN`) or the local HF CLI login.

## Datasets and their roles

| Dataset | Purpose |
|---|---|
| `karpathy/tiny_shakespeare` | Local smoke test — fast, no GPU rental required, validates the training loop end-to-end before spending money on a GPU. |
| `google/reformer-enwik8` | 15-minute A100 smoke test — validates loss/throughput numbers on real GPU hardware before committing to a full run. |
| `HuggingFaceFW/fineweb-edu` | Main pretraining corpus, run on rented RunPod A100. |
| `HuggingFaceTB/smoltalk` | SFT (supervised fine-tuning) after pretraining. |
| `HuggingFaceH4/no_robots` | Quick sanity checks (small, fast to iterate on). |

Expected workflow order: tiny_shakespeare smoke test (local) → reformer-enwik8 smoke test (A100, ~15 min) → fineweb-edu pretraining (A100) → smoltalk SFT, with no_robots available at any point for a fast sanity pass.

## Development principles

- **Fail-fast TDD**: write a failing test before writing the implementation; keep feedback loops short, especially given GPU rental costs make late-discovered bugs expensive.
- **SOLID**: apply standard SOLID design principles to the codebase (tokenizer, dataset loading, model, training loop, evaluation should be separable, substitutable concerns).
- **Karpathy principles for overengineering**: before adding abstraction, ask whether it earns its keep at the current scale of this toy project. Default to the simplest thing that works; prefer readable, hackable, single-purpose scripts over premature generalization. When in doubt about whether a component is overengineered, evaluate it against this standard rather than adding configurability "just in case."
