FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    DASH_HOST=0.0.0.0 \
    DASH_PORT=8050

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/

EXPOSE 8050

CMD ["gunicorn", "app.main:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--timeout", "60", \
     "--access-logfile", "-"]
