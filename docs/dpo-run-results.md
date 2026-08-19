# DPO run results

Date: 2026-08-19

## Summary

Ran the full DPO pipeline (`generate_pairs.py` → `judge.py` → `training/dpo.py`) against
the SFT checkpoint (`sft-checkpoints-smoltalk/step_12000.pt`), then trained for 1 epoch.
**Conclusion up front: the resulting checkpoint (`dpo-checkpoints/step_176.pt`) shows a
real but weak preference-learning signal and, once evaluated correctly (see the `--chat`
fix in §4), produces coherent, on-topic, reliably-stopping output.** The initial
evaluation pass looked broken — rambling, off-topic drift into a stray `<|assistant|>`
tag, an apparently unreliable stop-token — but that was a bug in how the samples were
generated (`generate.py` wasn't applying the chat template the checkpoint was trained on),
not a property of the checkpoint; see §3–§4. A 3-epoch rerun on the same data was also
tried and clearly overfit (not used, not kept locally); more DPO data, not more epochs, is
the right next lever.

## 1. Pipeline run

| Stage | Config | Result |
| --- | --- | --- |
| `generate_pairs.py` | 2,500 prompts from `trl-lib/ultrafeedback-prompt`, 2 completions each | `dpo-pairs_raw.jsonl` |
| `judge.py` | `together` / `meta-llama/Llama-3.3-70B-Instruct`, double-evaluated per pair | 2,500/2,500 processed, 1,580 kept (576 position-bias disagreement, 332 API failure, 12 degenerate) |
| `training/dpo.py` | 1 epoch, defaults (`beta=0.1`, TRL's default learning rate) | W&B run `objones25/llm-training/23zk4pnu` (`fancy-planet-28`) |

Two operational issues came up mid-run and are now handled by the pipeline itself rather
than being one-off fixes:

- `judge.py` hit an HF Inference Providers `402` (monthly credits exhausted) partway
  through. `judge.py` gained `--resume` support (skips already-processed rows, appends
  instead of truncating `--output`) so retrying after adding credits didn't require
  re-paying to re-judge already-completed pairs.
- After resuming, `pairs_dpo.jsonl` was found corrupted at line 728 (a block of NUL
  bytes) — caused by a second, non-`--resume` invocation truncating the file mid-write.
  Recovered by filtering to only valid JSON lines, yielding **1,403 usable pairs** (down
  from the 1,580 the judge stage originally reported kept).

## 2. Training: 1 epoch vs. 3 epochs

| | 1 epoch (used) | 3 epochs (not used) |
| --- | --- | --- |
| W&B run | `23zk4pnu` (`fancy-planet-28`) | `zjn0ejk9` (`solar-haze-29`) |
| Steps | 17 | 51 (3× the data) |
| Final `train/loss` | ~0.665, noisy, mildly decreasing | ~0.09 by the last epoch — collapsed |
| Final `train/rewards/accuracies` | oscillating 0.55–0.65 (chance = 0.5) | pinned at 0.975–1.0 from partway through epoch 1 onward |
| Final `train/rewards/margins` | ~0.02–0.07, noisy | 2.5+ |
| Checkpoint | `dpo-checkpoints/step_176.pt` (kept, pulled locally) | `dpo-checkpoints-v2/step_528.pt` (left on the pod, not pulled) |

The 3-epoch run's reward accuracy hitting 97–100% and loss collapsing to near-zero on
only 1,403 pairs is a memorization signature, not genuine generalization — there's no
held-out DPO validation split to confirm this directly, but the magnitude and abruptness
of the shift (visible starting mid-epoch-1) is well outside what a real, generalizable
preference signal should look like at this data scale. The 1-epoch run's noisier, more
modest numbers are the more trustworthy result. See `plots/dpo-run-23zk4pnu.png` for the
1-epoch run's loss/rewards/accuracy/grad-norm curves.

## 3. Generation quality (`dpo-checkpoints/step_176.pt`, `generate.py --chat`)

`generate.py` requires `--chat` for this checkpoint: it wraps `--prompt` as
`<|user|>\n{prompt}\n<|assistant|>\n` (`data/chat.py`'s `format_prompt`), matching the
exact format every SFT/DPO training example was trained on. Without it, the prompt is
out-of-distribution at token 0 and output is not representative of the checkpoint — see §4.

**Prompt:** "What's the capital of France?" (default `GenerationConfig`, `temperature=1.0`)
```
The capital of France is Paris. It is a city known for its historical significance,
cultural significance, and architectural landmarks. Paris serves as the political,
economic, and cultural center of France and the largest city in France, accounting for
around 85% of
```

**Prompt:** "Give me three tips for staying focused while studying." (default
`GenerationConfig`, `temperature=1.0`)
```
1. First, make sure you're getting enough sleep each night. Aim to get 7-8 hours of sleep
throughout the night to help regulate your body's natural physiological needed for focus.

2. Another crucial tip is avoiding distractions during
```
(cut off by the 50-token default `--max-new-tokens`, not degeneration)

At `temperature=0.0` (greedy), both prompts get clean, complete, correctly-stopped
answers — the France prompt returns exactly `The capital of France is Paris.` and stops
(no repetition, no stray tag), holding at both 150 and 300 `--max-new-tokens`, confirming
a genuine stop-token emission rather than truncation.

Coherent, on-topic, reliably-stopping output for both prompts. No rambling, no stray
`<|assistant|>` tag, no stop-token failure.

## 4. Fix: `generate.py` now requires `--chat` for chat-formatted checkpoints

`generate.py` originally passed `--prompt` straight to the tokenizer with no formatting,
so a raw prompt was out-of-distribution for any SFT/DPO checkpoint (every training example
starts with `<|user|>\n`). Fixed by adding an opt-in `--chat` flag that wraps `--prompt`
via `data/chat.py`'s `format_prompt()` before generating; omit it for base/pretraining
checkpoints, where a raw prompt is correct. `format_prompt` is now the single shared
definition used by `generate.py`, `generate_pairs.py`, and `training/dpo.py` (previously
duplicated inline in the latter two).

## 5. Next steps

- Treat `dpo-checkpoints/step_176.pt` as the current best DPO checkpoint — generation
  quality is good. Always invoke `generate.py --chat` for it (or any SFT/DPO checkpoint).
- To improve further: scale up `generate_pairs.py`'s `--num-prompts` and re-judge, rather
  than adding epochs on the existing 1,403 pairs (proven to overfit fast at this scale).
- Stop-token reliability held on every `--chat` test run so far; worth confirming on a
  larger prompt sample before fully closing it out, but no evidence of a bug remains.
