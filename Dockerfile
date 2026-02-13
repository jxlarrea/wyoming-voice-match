FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libsndfile1 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    # Uninstall triton (~600MB, not needed for inference)
    pip uninstall -y triton 2>/dev/null; \
    # Remove nccl (multi-GPU communication, not needed for single-GPU inference)
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/nccl && \
    # Strip PyTorch
    find /usr/local/lib/python3.10/dist-packages/torch -name "*.a" -delete && \
    find /usr/local/lib/python3.10/dist-packages/torch -name "test" -type d -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.10/dist-packages/torch -name "tests" -type d -exec rm -rf {} + 2>/dev/null; \
    cd /usr/local/lib/python3.10/dist-packages/torch/lib && \
    rm -f libnccl* libcublas* libcublasLt* libcusparse* libcusolver* \
          libcufft* libcurand* libnvrtc* libnvJitLink* libnvfuser* && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/distributed && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/_inductor && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/_dynamo && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/_functorch && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/ao && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/onnx && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/include && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/share && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/bin && \
    # Remove sympy and networkx (torch deps not needed at runtime for inference)
    pip uninstall -y sympy networkx 2>/dev/null; \
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/recipes && \
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/tests && \
    find /usr/local/lib/python3.10/dist-packages -name "tests" -type d -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.10/dist-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; \
    echo "Cleanup complete"

# --- Runtime stage ---
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

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

# Copy only the installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY wyoming_voice_match/ wyoming_voice_match/
COPY scripts/ scripts/

# Create data directory structure
RUN mkdir -p /data/enrollment /data/voiceprints /data/models

EXPOSE 10350

ENTRYPOINT ["python", "-m", "wyoming_voice_match"]