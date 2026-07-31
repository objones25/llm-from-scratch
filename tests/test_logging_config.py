import json
import logging

from llmtrain.logging_config import configure_logging


def test_configure_logging_writes_parsable_jsonl_with_extra_fields(tmp_path):
    log_file = tmp_path / "test.log"
    configure_logging(log_file=log_file)

    logger = logging.getLogger("llmtrain.test_logging_config")
    logger.info("order %s received", "abc123", extra={"order_id": "abc123"})
    logging.shutdown()

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["message"] == "order abc123 received"
    assert record["order_id"] == "abc123"
    assert "timestamp" in record
