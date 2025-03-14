FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Final image without uv
FROM python:3.10-slim-bookworm

WORKDIR /app

# Install make
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Run make pre-processing-pipeline
RUN make pre-processing-pipeline
