FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY config ./config
COPY src ./src
COPY "data/raw/Retail Datsets" "./data/raw/Retail Datsets"

RUN pip install --no-cache-dir -e .

ENV RETAIL_DATA_DIR="/app/data/raw/Retail Datsets" \
    RETAIL_ARTIFACTS_DIR="/app/artifacts" \
    RETAIL_PROCESSED_DIR="/app/data/processed" \
    RETAIL_OUTPUTS_DIR="/app/data/outputs" \
    RETAIL_LOG_LEVEL="INFO"

VOLUME ["/app/artifacts", "/app/data"]

ENTRYPOINT ["retail-analytics"]
CMD ["run-pipeline"]
