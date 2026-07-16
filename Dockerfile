# from pytorch/pytorch2.1.1-cuda12.1-cudnn8-runtime
# use above import for nvidian gpu support
FROM python:3.10-slim
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Optional extra pip args. When building behind a TLS-intercepting corporate
ARG PIP_EXTRA=""

COPY requirements.txt ./
RUN pip install $PIP_EXTRA --no-cache-dir --upgrade pip && \
    pip install $PIP_EXTRA --no-cache-dir -r requirements.txt pytest pygame

COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/
RUN pip install $PIP_EXTRA --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "-m"]
CMD ["pytest"]
