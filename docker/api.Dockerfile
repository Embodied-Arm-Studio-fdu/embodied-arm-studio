FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY agent ./agent
COPY apps ./apps
COPY courses ./courses
COPY migrations ./migrations
COPY robotics ./robotics
COPY templates ./templates
COPY scripts ./scripts
COPY simulation/src ./simulation/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]