FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git nano && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip
COPY requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt
WORKDIR /workspace
