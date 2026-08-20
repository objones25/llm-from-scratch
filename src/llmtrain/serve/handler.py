import logging
import os
from collections.abc import Iterator

from llmtrain.logging_config import configure_logging
from llmtrain.serve import generation
from llmtrain.training.config import DataConfig

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/runpod-volume/dpo-checkpoints/step_176.pt")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = generation.load_model_and_tokenizer(CHECKPOINT_PATH, TOKENIZER_PATH)
    return _model, _tokenizer


def handler(job: dict) -> Iterator[dict]:
    model, tokenizer = _get_model_and_tokenizer()
    payload = job.get("input", {})
    # RunPod job input is client-controlled; if it's ever a non-dict (a string, a
    # list, null) .get() below would raise an uncaught AttributeError outside the
    # structured-error path. Degrading to {} here means validate_messages() (via
    # stream_chat_completion) correctly rejects the resulting missing `messages` with
    # a clean structured error instead.
    if not isinstance(payload, dict):
        payload = {}
    messages = payload.get("messages")

    try:
        # parse_generation_config does bare int()/float() coercion on client-supplied
        # values -- a malformed payload (e.g. {"max_new_tokens": "abc"}) raises
        # ValueError/TypeError. Parsing it inside this try block (not before it) is
        # what routes that failure through the structured {"error": ..., "done": true}
        # response instead of letting it propagate uncaught.
        generation_cfg = generation.parse_generation_config(payload)
        for token_text in generation.stream_chat_completion(
            model, tokenizer, messages, generation_cfg, DataConfig.max_seq_len
        ):
            yield {"token": token_text, "done": False}
    except (ValueError, TypeError) as exc:
        logger.warning(f"rejected invalid request: {exc}")
        yield {"error": str(exc), "done": True}
        return

    yield {"done": True}


if __name__ == "__main__":
    import runpod

    configure_logging()
    logger.info(f"loading model and tokenizer from {CHECKPOINT_PATH}")
    _get_model_and_tokenizer()  # load once at process start, before accepting jobs
    logger.info("model loaded, starting RunPod serverless handler")
    # return_aggregate_stream=True makes /run and /runsync aggregate this generator
    # handler's yields into `output` too, not just /stream -- without it, confirmed
    # empirically during the live smoke test, /runsync returned output: [] even though
    # the handler ran correctly and /stream showed the real streamed tokens. The
    # website proxy uses /stream for SSE regardless, but this makes /run/, /runsync
    # usable for debugging, health checks, and curl testing without streaming
    # infrastructure.
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
