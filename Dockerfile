FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY apex/ ./apex/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e ".[all]"

# Verify installation
RUN python3 -c "from apex.model import APEX1Model; print('APEX-1 ready')"

# Default: run tests
CMD ["pytest", "tests/", "-v", "--tb=short"]
