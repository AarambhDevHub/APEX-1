
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY apex/ ./apex/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/ ./data/

RUN python -m pip install --upgrade pip \
    && pip install -e ".[dev,vision]"

RUN python -c "import apex; print('APEX-1', apex.__version__, 'ready')"

CMD ["python", "scripts/course_ready_check.py", "--mode", "quick"]
