# Inference API

A stateless, streaming chat completion API serving this project's DPO-tuned checkpoint
(`dpo-checkpoints/step_176.pt`), deployed on RunPod Serverless. See
[`docs/superpowers/specs/2026-08-19-inference-serving-design.md`](docs/superpowers/specs/2026-08-19-inference-serving-design.md)
for the full design rationale (why stateless, why no cross-request KV-cache, deployment
configuration) — this document is the request/response reference for calling it.

## Endpoint

```
https://api.runpod.ai/v2/kfk2n5xybfkdov
```

Deployed on RunPod Serverless (`US-IL-1`, scale-to-zero, `RTX A5000`-class GPU). A cold
start (first request after an idle period) takes roughly 100-120 seconds while a worker
spins up and loads the checkpoint; a warm request completes in about a second for a short
reply.

## Authentication

Every request needs a RunPod API key as a bearer token:

```
Authorization: Bearer <RUNPOD_API_KEY>
```

This key must never be embedded in client-side code — see "Integrating from a website"
below.

## Request

RunPod wraps the actual payload under an `input` key:

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "What's the capital of France?" }
    ],
    "max_new_tokens": 50,
    "temperature": 0.0
  }
}
```

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `messages` | array of `{role, content}` | yes | — | See Message rules below. |
| `max_new_tokens` | integer | no | `50` | Hard-capped server-side at **512** regardless of the requested value. |
| `temperature` | float | no | `1.0` | `0.0` = greedy decoding. Any value in `(0.0, 0.01)` is floored to `0.01` (prevents a near-zero value from crashing sampling); values above `2.0` are capped to `2.0`. |
| `top_p` | float | no | `1.0` | Clamped to `(0.0, 1.0]`; a value `<= 0.0` falls back to `1.0` (disabled). |
| `top_k` | integer | no | `0` (disabled) | Negative values are clamped to `0`. |
| `repetition_penalty` | float | no | `1.0` (no-op) | Clamped to `[1.0, 2.0]`. |

### Message rules

`messages` is validated before any model or GPU work runs — a violation returns a
structured error immediately (see Errors below), not a generation attempt.

- Must be a non-empty list, at most **50** entries.
- Each message needs `role` (`"user"` or `"assistant"`) and non-empty string `content`,
  at most **8000** characters.
- Roles must strictly alternate starting with `"user"`.
- The list must end on a `"user"` turn — the message the model is being asked to reply to.

### Statelessness — resend the full history every call

The API has no server-side session state. Every request must include the *entire*
conversation so far, not just the newest message:

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "What's the capital of France?" },
      { "role": "assistant", "content": "The capital of France is Paris." },
      { "role": "user", "content": "And Germany?" }
    ]
  }
}
```

If the formatted history plus `max_new_tokens` would exceed the model's 2048-token
context window, the **oldest** user/assistant turn-pairs are silently dropped from the
front (never mid-turn) until it fits — the caller doesn't need to track this itself. The
one exception: if even the single newest message alone doesn't fit, there's nothing left
to truncate, and that request gets a structured error instead (see Errors).

## Response

The handler streams tokens as they're generated. Each chunk is one JSON object; the
stream ends with `{"done": true}`:

```json
{"token": "The", "done": false}
{"token": " capital", "done": false}
{"token": " of", "done": false}
{"token": " France", "done": false}
{"token": " is", "done": false}
{"token": " Paris", "done": false}
{"token": ".", "done": false}
{"token": "\n", "done": false}
{"done": true}
```

Concatenating every `token` in order gives the full reply text.

### Errors

Invalid input (bad `messages`, malformed sampling parameters) never crashes the job — it
returns one structured chunk instead of streaming any tokens:

```json
{"error": "messages must be a non-empty list", "done": true}
```

## Calling it

RunPod exposes three ways to invoke a job; pick based on whether you want to stream.

### Streaming (recommended for a chat UI)

Submit with `/run`, then read the tokens as they arrive from `/stream`:

```bash
JOB_ID=$(curl -s -X POST "https://api.runpod.ai/v2/kfk2n5xybfkdov/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "What'\''s the capital of France?"}], "max_new_tokens": 50, "temperature": 0}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -s "https://api.runpod.ai/v2/kfk2n5xybfkdov/stream/$JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

A browser-facing proxy re-emits this stream as Server-Sent Events to the client (see
"Integrating from a website" below).

### Non-streaming (debugging, health checks, curl one-liners)

`/runsync` waits for the job to finish and returns the full token list in one response
(capped at RunPod's own gateway wait limit; falls back to an `IN_QUEUE`/`IN_PROGRESS`
status if the job outlives it — poll `/status/{job_id}` in that case):

```bash
curl -s -X POST "https://api.runpod.ai/v2/kfk2n5xybfkdov/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "What'\''s the capital of France?"}], "max_new_tokens": 50, "temperature": 0}}'
```

```json
{
  "id": "sync-...",
  "status": "COMPLETED",
  "output": [
    {"token": "The", "done": false},
    {"token": " capital", "done": false},
    "...",
    {"done": true}
  ]
}
```

## Integrating from a website

The RunPod API key must never be embedded in client-side (browser) code — a static site
needs a thin server-side proxy (an edge/serverless function) that holds the key, forwards
the client's `messages` to this endpoint, and re-emits the response as SSE. That proxy is
website-specific and lives in the website's own repo, not this one; this document defines
the contract it needs to satisfy against this API.

## Model notes

- Checkpoint: `dpo-checkpoints/step_176.pt` — the current best DPO checkpoint (1 epoch,
  1,403 judged preference pairs). See the main [`README.md`](README.md) for example
  output and its documented quality ceiling.
- Trained context window: 2048 tokens.
- No cross-request KV-cache — each request re-runs the full prefill. See the design
  spec's "Caching rationale" section for why this is the right tradeoff at this model's
  size and traffic level.
