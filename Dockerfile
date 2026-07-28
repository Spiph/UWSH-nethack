# syntax=docker/dockerfile:1.7
FROM python:3.10.14-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential bison cmake flex git ninja-build libbz2-dev libffi-dev liblzma-dev \
    libncurses5-dev libsqlite3-dev libssl-dev pkg-config zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /usr/local/bin/uv
WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
# The SOL compatibility profile carries its patched NLE and Sample Factory fork
# together, matching the upstream SOL checkout rather than mixing incompatible
# release pins.
RUN git clone --filter=blob:none https://github.com/facebookresearch/sol.git /opt/sol \
    && git -C /opt/sol checkout "7c272b66e6ebe72ca008526d33f7e2e40e660af5" \
    && uv sync --frozen --extra sol --extra adapter --no-dev --no-install-project \
    && cd /opt/sol/sample_factory/algo/utils/cython \
    && /workspace/.venv/bin/python setup.py build_ext --inplace \
    && cd /workspace \
    && uv pip install --python /workspace/.venv/bin/python --no-deps -e /opt/sol \
    && uv pip install --python /workspace/.venv/bin/python --no-deps -e /opt/sol/nle_patched
COPY src ./src
RUN uv sync --frozen --extra sol --extra adapter --no-dev

FROM python:3.10.14-slim-bookworm AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libncurses6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /workspace /workspace
COPY --from=build /opt/sol /opt/sol
WORKDIR /workspace
ENV PATH="/workspace/.venv/bin:${PATH}" PYTHONPATH="/opt/sol:${PYTHONPATH}" \
    SOL_COMMIT="7c272b66e6ebe72ca008526d33f7e2e40e660af5" UPS_CONTAINER_IMAGE="ups:cpu"
ENTRYPOINT ["ups"]
CMD ["--help"]

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 libglib2.0-0 libncurses6 \
    && ln -s /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /workspace /workspace
COPY --from=build /opt/sol /opt/sol
WORKDIR /workspace
ENV PATH="/workspace/.venv/bin:${PATH}" PYTHONPATH="/opt/sol:${PYTHONPATH}" \
    SOL_COMMIT="7c272b66e6ebe72ca008526d33f7e2e40e660af5" UPS_CONTAINER_IMAGE="ups:gpu" NVIDIA_VISIBLE_DEVICES="all"
ENTRYPOINT ["ups"]
CMD ["--help"]
