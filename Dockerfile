# from pytorch/pytorch2.1.1-cuda12.1-cudnn8-runtime
# use above import for nvidian gpu support 
FROM python:3.10-slim
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py READMENE.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m"]
