FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential wget ca-certificates tzdata &&     rm -rf /var/lib/apt/lists/*

# TensorFlow CPU + libs (on IoT; edges/cloud analyze don't need TF but it's fine)
# Keep versions moderate for compatibility
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 websockets==12.0     pyyaml==6.0.2 matplotlib==3.9.0 scikit-learn==1.5.1     tensorflow-cpu==2.15.0.post1

WORKDIR /workspace

# Analyzer + data prep
COPY project /workspace/project
COPY data /workspace/data
COPY config /workspace/config
RUN mkdir -p /workspace/outputs /workspace/logs
