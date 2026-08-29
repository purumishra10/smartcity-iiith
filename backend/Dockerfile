FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend /app/backend
COPY ml/artifacts /app/ml/artifacts

ENV PYTHONPATH=/app/backend
ENV DATABASE_URL=postgresql+psycopg2://clinic:clinic@postgres:5432/clinic
ENV STORAGE_DIR=/data/storage
ENV MODEL_PATH=/app/ml/artifacts/model.pt
ENV SCALER_PATH=/app/ml/artifacts/scaler.json

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
