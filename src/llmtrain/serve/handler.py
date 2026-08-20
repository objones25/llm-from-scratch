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
    messages = payload.get("messages")
    generation_cfg = generation.parse_generation_config(payload)

    try:
        for token_text in generation.stream_chat_completion(
            model, tokenizer, messages, generation_cfg, DataConfig.max_seq_len
        ):
            yield {"token": token_text, "done": False}
    except ValueError as exc:
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
    runpod.serverless.start({"handler": handler})
