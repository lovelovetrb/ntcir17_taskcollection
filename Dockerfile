FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# UV_PROJECT_ENVIRONMENT: /work はホストをバインドマウントするため venv を外に置く
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/

WORKDIR /work

# .python-version が無いと uv が requires-python を満たす最新版を選んでしまう
COPY pyproject.toml uv.lock .python-version ./
RUN uv python install && uv sync --all-groups --no-install-project

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

CMD ["bash"]
