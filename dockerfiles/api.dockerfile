FROM python:3.11-slim AS base
EXPOSE 8080

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl wget && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts && \
    ln -s /google-cloud-sdk/bin/gsutil /usr/bin/gsutil

RUN /root/google-cloud-sdk/bin/gsutil cp gs://embedding_data_mlops/train_embeddings.pkl train_embeddings.pkl

COPY src/mlops_project/api.py src/mlops_project/api.py
COPY src/mlops_project/model.py src/mlops_project/model.py

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

CMD ["uvicorn", "src.mlops_project.api:app", "--port", "8080", "--host", "0.0.0.0", "--workers", "1"]
