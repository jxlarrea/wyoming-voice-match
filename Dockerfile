FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libsndfile1 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --no-cache-dir torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128; \
    else \
        pip install --no-cache-dir --pre torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128; \
    fi && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$TARGETARCH" != "amd64" ]; then \
        pip install --no-cache-dir --upgrade speechbrain; \
    fi && \
    pip uninstall -y triton 2>/dev/null; \
    if [ "$TARGETARCH" = "amd64" ]; then \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cublas && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cudnn && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cufft && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/curand && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cusolver && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cusparse && \
        echo "Keeping NCCL (required by libtorch_cuda.so)" && \
        rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink && \
        cd /usr/local/lib/python3.10/dist-packages/torch/lib && \
        rm -f libcublas* libcublasLt* libcusolver* \
              libcufft* libcurand* libnvrtc* libnvJitLink* libnvfuser*; \
    fi && \
    find /usr/local/lib/python3.10/dist-packages/torch -name "*.a" -delete && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/include && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/share && \
    pip uninstall -y sympy networkx 2>/dev/null; \
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/recipes && \
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/tests && \
    find /usr/local/lib/python3.10/dist-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; \
    echo "Cleanup complete"

# --- Runtime stage ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="Wyoming Voice Match"
LABEL description="Wyoming ASR proxy with ECAPA-TDNN speaker verification"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        libsndfile1 \
        ffmpeg \
        libgomp1 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY wyoming_voice_match/ wyoming_voice_match/
COPY scripts/ scripts/

# Patch SpeechBrain's torchaudio backend check (list_audio_backends removed in newer torchaudio)
RUN sed -i 's/available_backends = torchaudio.list_audio_backends()/available_backends = []/' \
        /usr/local/lib/python3.10/dist-packages/speechbrain/utils/torch_audio_backend.py

RUN mkdir -p /data/enrollment /data/voiceprints /data/models

EXPOSE 10350
ENTRYPOINT ["python", "-m", "wyoming_voice_match"]