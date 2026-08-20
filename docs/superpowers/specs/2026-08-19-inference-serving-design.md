# Inference Serving (RunPod Serverless) — Design

Date: 2026-08-19

## Scope

Adds a public-facing inference API for the current best DPO checkpoint
(`dpo-checkpoints/step_176.pt`), deployed on RunPod Serverless, to power a chat demo embedded on
the user's personal website.

**In scope:**

- A RunPod serverless handler that streams multi-turn chat completions from the existing
  checkpoint, reusing the existing KV-cache generation code rather than reimplementing it.
- A small addition to `data/chat.py` (`format_chat_history`) to format a multi-turn message list,
  since the existing `format_prompt`/`format_turn` only handle a single user turn today.
- The API request/response contract, context-window overflow policy, and RunPod deployment
  configuration (GPU tier, scaling, checkpoint loading).

**Explicitly deferred / excluded:**

- **The website's own proxy/edge function.** A static site can't hold the RunPod API key
  client-side, so a thin serverless function on the website side is required to auth-gate,
  rate-limit, and forward requests — but that lives in the website's own repo/stack, not this one.
  This design only documents the contract that proxy must satisfy.
- **Cross-request KV-cache persistence (e.g. Redis).** Considered and rejected — see Caching
  rationale below.
- **Always-warm workers (`min workers > 0`).** Rejected in favor of scale-to-zero; a personal demo
  with sporadic traffic doesn't justify paying for continuously-running GPU time, and a 10-60s cold
  start on the first message after an idle period is an acceptable tradeoff.
- **Server-side conversation sessions.** The API is stateless — the client sends the full message
  history on every call, matching how RunPod serverless workers are provisioned (not sticky across
  a conversation) and keeping the handler a pure function with nothing to lose on a cold start.
- **DPO pipeline improvements.** Considered as an alternative use of this time/budget and
  deliberately deprioritized: the DPO checkpoint's reward accuracy (0.55-0.65, chance = 0.5) is a
  real but weak signal, and scaling preference-pair count further would tighten that signal
  somewhat but can't move the model's actual ceiling, since the base model itself was diagnosed as
  capacity-limited rather than undertrained. Building a way for anyone to actually use the existing
  pipeline's output is the better use of the next chunk of time/money than incremental DPO polish.

## Motivation

The full pretrain → SFT → DPO pipeline works end to end and produces a checkpoint worth showing
off, but nothing currently lets anyone but the project owner run it — there's no serving layer at
all. Given the DPO checkpoint's quality ceiling is already bounded by base-model capacity (not by
DPO pair count), the better return on the next investment of time/money is making the existing
result usable and visible, not chasing marginal DPO gains. A RunPod-serverless-backed demo on the
personal website is also a different kind of payoff than model quality: it showcases the whole
pipeline (data engineering, custom transformer, tokenizer, training loop, DPO), which is the more
interesting engineering story for this project anyway.

## Architecture

New `src/llmtrain/serve/` module:

```text
src/llmtrain/serve/
  generation.py   # RunPod-SDK-free core: loads checkpoint + tokenizer once at import time,
                  # exposes stream_chat_completion(messages, generation_config) -> Iterator[str]
  handler.py      # thin RunPod SDK adapter (runpod.serverless.start) -- parses job input,
                  # calls stream_chat_completion, yields RunPod's streaming chunk shape.
                  # No model logic lives here.
```

Keeping `generation.py` free of any RunPod SDK dependency means it's unit-testable (with mocks)
and directly callable from a local script, without needing a deployed endpoint or the `runpod`
package installed — `handler.py` is the only file that knows it's running on RunPod.

**Gap this surfaces**: `data/chat.py`'s `format_prompt()` only wraps a single user turn today
(`format_turn("user", prompt) + "<|assistant|>\n"`), which is all `generate.py`/`generate_pairs.py`
have ever needed. Multi-turn chat needs a new `format_chat_history(messages: list[dict]) -> str`
that loops `format_turn(role, content)` over the full message list and appends the trailing
`<|assistant|>\n` — a small addition that reuses `format_turn`, not a new formatting
implementation.

**Checkpoint loading**: the container mounts the RunPod network volume directly at its native
serverless mount path, rather than downloading through `s3.py`'s S3-compatible-API path — this
skips a download hop on every cold start. `--checkpoint`/`--tokenizer-path` just point at the
mounted path. `s3.py`'s S3 path remains available for local dev against a stopped pod, unchanged.

## Caching rationale (why no cross-request KV cache)

Considered storing KV-cache state in Redis so a conversation's second-and-later turns wouldn't
need to re-run the full prefill. Rejected for three reasons:

1. **Size.** At the current architecture (`n_layers=20`, `n_kv_heads=4`, `head_dim=72`), the cache
   is `2 x 20 x 4 x 72 = 11,520` floats/token, ~22.5KB/token at fp16 — ~11MB for a 500-token
   conversation, ~46MB near the 2048-token context ceiling. Serializing that to Redis, shipping it
   over the network, and deserializing it back onto GPU memory each turn plausibly costs more than
   just recomputing the prefill.
2. **RunPod serverless workers aren't sticky.** Turn 2 of a conversation isn't guaranteed to land
   on the same worker as turn 1, and a scale-to-zero worker will likely be gone entirely by the
   time a real user sends their next message anyway (personal-demo traffic is sporadic). Redis
   would mostly miss regardless of the serialization cost above.
3. **The thing it would save is cheap at this model's scale.** 478.6M params — prefill over a few
   hundred to ~1500 tokens is low tens of milliseconds on an A100-class GPU. This isn't a large
   model where prefill dominates request cost.

`model/cache.py`'s `KVCache` stays exactly as it is: built fresh per request, used only to make
that one request's own decode loop efficient. Stateless, recompute-prefill-per-turn is the
simplest thing that works and nothing about this project's scale argues for more.

## API contract

**Request** (RunPod job input):

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "max_new_tokens": 150,
  "temperature": 0.7
}
```

- `messages`: required, non-empty, must end on a `user` turn, roles must alternate `user`/
  `assistant`. Validated in `generation.py` before any model/GPU work, so malformed input fails
  fast with a structured error.
- `max_new_tokens`/`temperature`/`top_k`/`top_p`/`repetition_penalty`: optional, default from the
  existing `GenerationConfig`. `max_new_tokens` additionally gets a hard server-side ceiling
  (e.g. 512) regardless of what the client requests, since this is a public-facing endpoint and
  generation time is direct cost exposure.

**Response**: streamed chunks via RunPod's generator-handler support, each roughly
`{"token": "...", "done": false}`, ending with `{"done": true}`. The website's proxy re-emits
these as SSE to the browser.

**Context-window overflow policy**: `messages` are formatted via `format_chat_history` and
tokenized; if `len(formatted_history) + max_new_tokens` exceeds the model's trained context
(`max_seq_len=2048`), the oldest user/assistant turn-pairs are truncated from the front (never
mid-turn) until it fits, rather than rejecting the request. Matches how most chat UIs silently
manage context, and keeps the website's proxy free of special-case error handling for it. This
policy is about *conversation history*, not about a single oversized message: if even the final
(newest) user turn alone still doesn't fit — a user pasting an enormous amount of text in one
message — there is no history left to truncate, and that case *does* surface as a structured
`{"error": ..., "done": true}` chunk via the existing bad-input error path, rather than silently
truncating the user's own message content (which would drop part of what they wrote without
telling them). A rare edge case, not a violation of the policy above.

## Deployment configuration

- **Network volume**: `dpo-checkpoints/step_176.pt` + `tokenizer.json` live on a dedicated 10GB
  network volume (`US-IL-1`, `STANDARD` tier). The serverless endpoint must be deployed in the same
  data center as this volume for the mount to work — and, confirmed the hard way during
  deployment, a data center supporting network volumes / the S3-compatible API is **not** the same
  as one having any GPU serverless compute at all: the volume originally landed in `US-MD-1`
  (S3-API-enabled, real checkpoint uploads worked fine there) only to discover `US-MD-1` has zero
  GPU serverless capacity for every GPU type — `get-gpu-type`'s per-data-center availability list
  never included it. The volume (and its contents) had to be migrated to `US-IL-1`, which has both
  S3-API access and real GPU stock. Check a candidate data center against both requirements before
  provisioning a volume there, not just the S3-API-enabled list.
- **Image platform**: the container image must be built for `linux/amd64` explicitly
  (`docker build`/`buildx build --platform linux/amd64`) — RunPod's GPU workers are all x86_64, but
  `docker build` defaults to the host machine's architecture, so building on Apple Silicon without
  the flag silently produces a `linux/arm64` image that would fail to run on any RunPod worker.
- **GPU tier**: cheapest RunPod serverless tier with enough VRAM. Model weights are
  ~478.6M params x 2 bytes (fp16) ~= 0.95GB, plus per-request KV cache (tens of MB even near the
  2048-token ceiling) and framework overhead — comfortably fits RunPod's smallest GPU tier
  (16GB class). No reason to reach for a bigger tier.
- **Scaling**: `min workers = 0` (scale-to-zero), `max workers` capped low (e.g. 2-3) — bounds
  worst-case concurrent cost from any traffic burst; a personal demo has no legitimate need for
  more concurrency than that.
- **Execution timeout**: set generously above worst-case generation time at `max_new_tokens=512`,
  but still bounded, so a stuck/hung job can't run indefinitely.

## Error handling

- **Bad input** (empty/malformed `messages`): validated before model touch; structured error
  returned through the job result with no GPU cost incurred.
- **Context overflow**: handled silently via the truncation policy above, not an error path.
- **Generation-time failure** (CUDA OOM, unexpected exception): job marked failed by RunPod's SDK;
  logged via the existing JSONL `logging_config.py`; generic error surfaced to the proxy.
- **Cold-start failure** (checkpoint missing/corrupt at the mounted volume path): fails at
  container import time, so RunPod reports the endpoint unhealthy rather than silently serving
  broken requests.

## Testing

Same principle as the rest of the repo: everything except the GPU decode loop itself gets a fast,
CPU-only unit test with tiny fake data.

| Component | Tested with tiny fake data / mocks | Not automated (manual smoke test) |
| --- | --- | --- |
| `data/chat.py::format_chat_history` | pure string formatting over a small fixed message list | — |
| Context-overflow truncation logic | small fake `max_seq_len`, asserts oldest-turn-pairs dropped, never mid-turn | — |
| `serve/generation.py` request validation | malformed/empty/non-alternating `messages` -> structured error, before any model call | — |
| `serve/handler.py` | RunPod-adapter glue, tested by mocking `stream_chat_completion` — no RunPod SDK or GPU required | — |
| End-to-end generation quality | — | manual smoke test against the real deployed endpoint (same pattern as the existing README smoke test), since it needs a real GPU + real checkpoint |

## Open questions carried into the implementation plan

- Exact RunPod generator-handler streaming API shape to confirm against current RunPod SDK docs
  at implementation time (chunk format, how `done` is signaled, error-mid-stream behavior).
- Exact `max_new_tokens` ceiling and `max workers`/execution-timeout values — reasonable defaults
  proposed above, to be tuned once real generation latency on the target GPU tier is measured.
- The website proxy's exact contract (auth check, rate-limit granularity, SSE re-emission) is
  deferred to the website's own project, but its interface with this API should be confirmed
  before that side is built.
