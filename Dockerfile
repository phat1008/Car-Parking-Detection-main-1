FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/parking_cnn.keras \
    CONFIG_PATH=/app/config/parking_config.json

WORKDIR /app

COPY web/requirements-web.txt /app/web/requirements-web.txt
RUN pip install --no-cache-dir -r /app/web/requirements-web.txt

COPY src /app/src
COPY parking_detector.py /app/parking_detector.py
COPY config /app/config
COPY models/parking_cnn.keras models/parking_cnn.metadata.json /app/models/
COPY web /app/web

# Render cấp biến PORT; mặc định 8000 khi chạy local.
EXPOSE 8000
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:${PORT:-8000} 'web.app:create_app()'"]
