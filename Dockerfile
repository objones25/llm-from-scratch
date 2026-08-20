# RunPod's GPU workers are all x86_64. `docker build` defaults to the host machine's
# architecture, so building this image on Apple Silicon without an explicit platform
# flag silently produces a linux/arm64 image that fails to run on any RunPod worker --
# this bit the actual deployment once already, see
# docs/superpowers/specs/2026-08-19-inference-serving-design.md's Deployment
# configuration section. Always build with:
#   docker buildx build --platform linux/amd64 -t <tag> .
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv python install 3.12 && \
    uv sync --frozen --extra serve --extra s3 --no-dev

ENV CHECKPOINT_PATH=/runpod-volume/dpo-checkpoints/step_176.pt
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "-m", "llmtrain.serve.handler"]
