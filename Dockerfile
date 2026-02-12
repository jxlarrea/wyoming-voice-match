FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Wyoming Voice Match"
LABEL description="Wyoming ASR proxy with ECAPA-TDNN speaker verification"

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libsndfile1 \
        ffmpeg && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 12.1) with minimal footprint, then app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY wyoming_voice_match/ wyoming_voice_match/
COPY scripts/ scripts/

# Create data directory structure
RUN mkdir -p /data/enrollment /data/voiceprints /data/models

EXPOSE 10350

ENTRYPOINT ["python", "-m", "wyoming_voice_match"]
