FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
# Resolve the production model by name from the MLflow registry. Point
# MLFLOW_TRACKING_URI at the registry store and name the model to serve.
ENV TURBOFAN_MODEL_NAME=turbofan-gru-fd001
ENV TURBOFAN_MODEL_ALIAS=production
ENV MLFLOW_TRACKING_URI=sqlite:////models/mlflow.db

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# Install CPU-only torch first to avoid pulling the ~2 GB CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["turbofan-serve-api", "--host", "0.0.0.0", "--port", "8000"]
