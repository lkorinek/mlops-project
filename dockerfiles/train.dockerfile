# Base image
#FROM  nvcr.io/nvidia/pytorch:22.07-py3 AS base
FROM python:3.11-slim AS base

ARG DEFAULT_JSON
RUN printf "%s" "${DEFAULT_JSON}" > default.json

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY configs configs/
RUN mkdir -p models
RUN mkdir -p reports/figures
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

# data
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY data.dvc data.dvc
RUN dvc config core.no_scm true

RUN dvc remote modify --local remote_storage credentialpath default.json
RUN dvc pull --no-run-cache

ENTRYPOINT ["python", "-u", "src/mlops_project/train.py"]
