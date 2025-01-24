FROM python:3.11-slim AS base
EXPOSE $PORT

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Not at all best practice, but due to deadline, use a service account key
COPY service_account.json service_account.json
# Need this to access google cloud storage on docker
ENV GOOGLE_APPLICATION_CREDENTIALS="service_account.json"
# Files we need
COPY train_embeddings.pkl train_embeddings.pkl
COPY src/mlops_project/api.py src/mlops_project/api.py
COPY src/mlops_project/model.py src/mlops_project/model.py

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

CMD ["uvicorn", "src.mlops_project.api:app", "--port", "$PORT", "--host", "0.0.0.0", "--workers", "1"]
