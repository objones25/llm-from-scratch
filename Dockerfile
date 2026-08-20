FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv python install 3.12 && \
    uv sync --frozen --extra cuda --extra serve --extra s3 --no-dev

ENV CHECKPOINT_PATH=/runpod-volume/dpo-checkpoints/step_176.pt
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "-m", "llmtrain.serve.handler"]
