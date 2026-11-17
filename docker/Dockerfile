ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}
ARG PYTHON_BIN=/usr/local/bin/python
ARG TRAIN_ENV=/opt/frozenlake-venv
USER root
RUN ${PYTHON_BIN} -m ensurepip --upgrade \
    && ${PYTHON_BIN} -m pip install --no-cache-dir --ignore-installed --prefix=/opt/uv 'uv==0.13.8'
ENV PATH="/opt/uv/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT=${TRAIN_ENV}
WORKDIR /opt/frozenlake-project
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group train --no-install-project --python ${PYTHON_BIN} --no-python-downloads
ENV FROZENLAKE_PYTHON=${TRAIN_ENV}/bin/python
COPY docker/entrypoint.sh /opt/frozenlake-entrypoint
RUN chmod 755 /opt/frozenlake-entrypoint
ENV HOME=/tmp
ENV PYTHONPATH=/ws/src
ENV JAX_ENABLE_X64=true
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /ws
USER 1001:1001
ENTRYPOINT ["/opt/frozenlake-entrypoint"]
