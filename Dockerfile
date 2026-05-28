FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system packages needed by some Python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Put the venv outside /app so the host bind-mount doesn't shadow it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install deps (cached layer — only rebuilds when pyproject.toml or uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy source for standalone `docker build` / `docker run` use.
# At runtime the whole repo is bind-mounted over /app, so this layer is just
# a fallback for CI or bare docker run.
COPY . .
