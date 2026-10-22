FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app:/app/simulation/src
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/simulation-requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir pip==25.0.1 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check
COPY simulation ./simulation
COPY robotics ./robotics
