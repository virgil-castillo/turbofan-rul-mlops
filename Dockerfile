FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV TURBOFAN_MODEL_ARTIFACT=/models/model_manifest.json

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["turbofan-serve-api", "--host", "0.0.0.0", "--port", "8000"]
