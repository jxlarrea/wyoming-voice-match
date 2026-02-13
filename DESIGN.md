# Design & Architecture

Complete technical specification for Wyoming Voice Match. This document contains enough detail for a developer (or LLM) to rebuild the project from scratch.

## Project Structure

```
wyoming-voice-match/
├── wyoming_voice_match/          # Main Python package
│   ├── __init__.py               # Version string (__version__ = "1.0.0")
│   ├── __main__.py               # Entry point, arg parsing, server setup
│   ├── handler.py                # Wyoming event handler (ASR proxy logic)
│   └── verify.py                 # ECAPA-TDNN speaker verification engine
├── scripts/
│   ├── __init__.py               # Empty, makes scripts a package
│   ├── enroll.py                 # Voice enrollment CLI
│   └── test_verify.py            # Threshold tuning CLI
├── tools/
│   └── record_samples.ps1        # Windows PowerShell recording helper
├── Dockerfile                    # GPU image (CUDA 12.4 + cuDNN, multi-stage)
├── Dockerfile.cpu                # CPU image (python:3.11-slim)
├── docker-compose.yml            # GPU compose config
├── docker-compose.cpu.yml        # CPU compose config
├── requirements.txt              # GPU Python deps (torch installed separately via cu121 index)
├── requirements.cpu.txt          # CPU Python deps (torch from default PyPI)
├── LICENSE                       # MIT
└── README.md                     # User-facing documentation
```

## Dependencies

### Python Packages

**requirements.txt** (GPU — torch/torchaudio installed separately from `https://download.pytorch.org/whl/cu121`):
```
wyoming==1.8.0
speechbrain>=1.0.0
scipy>=1.11.0
numpy>=1.24.0
requests>=2.28.0
huggingface_hub<0.27.0
```

**requirements.cpu.txt** (includes torch from default PyPI):
```
wyoming==1.8.0
speechbrain>=1.0.0
torch>=2.1.0
torchaudio>=2.1.0
scipy>=1.11.0
numpy>=1.24.0
requests>=2.28.0
huggingface_hub<0.27.0
```

Key dependency notes:
- `huggingface_hub<0.27.0` — pinned to avoid a breaking change in 0.27+ that affects SpeechBrain's model loading
- `wyoming==1.8.0` — the Wyoming protocol library (AsyncServer, AsyncEventHandler, AsyncClient, event types)
- `speechbrain` — provides `EncoderClassifier` for ECAPA-TDNN inference
- `scipy` — only used for `scipy.spatial.distance.cosine`
- `torchaudio` — only used in enrollment/test scripts for loading WAV files (the main service receives raw PCM bytes)

### System Packages (in Docker)

- `libsndfile1` — required by SpeechBrain/torchaudio for audio I/O
- `ffmpeg` — used by enrollment script for format conversion
- `libgomp1` — OpenMP runtime (GPU image only, required by torch)

## Docker Images

### GPU Image (Dockerfile)

Multi-stage build to minimize image size (~5GB):

**Stage 1 (builder):** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- Installs python3, python3-pip, python3-dev, libsndfile1
- Installs torch and torchaudio from `https://download.pytorch.org/whl/cu121`
- Installs remaining requirements from requirements.txt
- Cleanup step removes ~2GB of redundant files:
  - Triton (~600MB, not needed for inference)
  - Duplicate NVIDIA pip packages (cublas, cuda_runtime, cudnn, etc.) that are already in the CUDA base image
  - Duplicate CUDA libs from torch/lib (keeps cusparseLt)
  - Static libs (.a files), include dirs, share dirs
  - Unused pip packages (sympy, networkx)
  - SpeechBrain recipes and tests directories
  - All `__pycache__` directories

**Stage 2 (runtime):** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- Installs python3, libsndfile1, ffmpeg, libgomp1
- Copies installed Python packages from builder
- Copies application code (wyoming_voice_match/ and scripts/)
- Creates /data directory structure (enrollment, voiceprints, models)
- Entrypoint: `python -m wyoming_voice_match`

### CPU Image (Dockerfile.cpu)

Single-stage, much simpler: `python:3.11-slim`
- Installs libsndfile1, ffmpeg
- Installs all Python deps from requirements.cpu.txt (CPU-only torch from PyPI)
- Same application code and directory structure
- Same entrypoint

### Data Volume

The `/data` directory is mounted as a Docker volume and contains:
- `/data/enrollment/<speaker>/` — WAV files for each enrolled speaker
- `/data/voiceprints/<speaker>.npy` — generated voiceprint embeddings (192-dim numpy arrays)
- `/data/models/spkrec-ecapa-voxceleb/` — cached ECAPA-TDNN model from HuggingFace
- `/data/hf_cache/` — HuggingFace Hub cache (set via `HF_HOME` env var)

## Entry Point (__main__.py)

### Argument Parsing

All arguments have environment variable fallbacks for Docker configuration:

| CLI Flag | Env Var | Default | Description |
|---|---|---|---|
| `--uri` | `LISTEN_URI` | `tcp://0.0.0.0:10350` | Wyoming server listen URI |
| `--upstream-uri` | `UPSTREAM_URI` | `tcp://localhost:10300` | Upstream ASR service URI |
| `--voiceprints-dir` | `VOICEPRINTS_DIR` | `/data/voiceprints` | Directory with .npy voiceprints |
| `--threshold` | `VERIFY_THRESHOLD` | `0.20` | Cosine similarity threshold |
| `--device` | `DEVICE` | `cuda` | `cuda` or `cpu` |
| `--model-dir` | `MODEL_DIR` | `/data/models` | Model cache directory |
| `--debug` | `LOG_LEVEL=DEBUG` | `INFO` | Enable debug logging |
| `--max-verify-seconds` | `MAX_VERIFY_SECONDS` | `5.0` | Early verification trigger |
| `--window-seconds` | `VERIFY_WINDOW_SECONDS` | `3.0` | Sliding window size |
| `--step-seconds` | `VERIFY_STEP_SECONDS` | `1.5` | Sliding window step |
| `--asr-max-seconds` | `ASR_MAX_SECONDS` | `3.0` | Max audio sent to ASR |

### Startup Sequence

1. Parse args
2. Configure logging (DEBUG or INFO)
3. Validate voiceprints directory exists (exit 1 if not)
4. Create `SpeakerVerifier` — loads ECAPA-TDNN model and all .npy voiceprints
5. Validate at least one voiceprint loaded (exit 1 if not)
6. Build `wyoming.info.Info` with ASR program/model metadata
7. Create `AsyncServer` and run with `SpeakerVerifyHandler` factory

The handler factory uses `functools.partial` to pass `wyoming_info`, `verifier`, `upstream_uri`, and `asr_max_seconds` to each new handler instance.

### Wyoming Service Info

The service registers as an ASR program named `"voice-match"` with model `"voice-match-proxy"`, language `["en"]`. This makes it appear as a standard STT service in Home Assistant.

## Handler (handler.py)

### Class: SpeakerVerifyHandler

Extends `wyoming.server.AsyncEventHandler`. One instance per TCP connection.

**Module-level state:**
- `_MODEL_LOCK: asyncio.Lock` — prevents concurrent ECAPA-TDNN inference across all handlers
- `_ASR_PADDING_SEC = 0.5` — extra seconds before detected speech start (not currently used in trimming, kept for future use)

**Instance state:**
- `wyoming_info: Info` — service metadata for Describe responses
- `verifier: SpeakerVerifier` — shared verifier instance
- `upstream_uri: str` — ASR service URI
- `asr_max_seconds: float` — max audio forwarded to ASR
- `_audio_buffer: bytes` — accumulated PCM audio
- `_audio_rate: int` — sample rate (default 16000)
- `_audio_width: int` — bytes per sample (default 2 = 16-bit)
- `_audio_channels: int` — channel count (default 1 = mono)
- `_language: Optional[str]` — language from Transcribe event
- `_verify_task: Optional[asyncio.Task]` — background verification task
- `_verify_started: bool` — whether early verification was triggered
- `_responded: bool` — whether transcript was already sent
- `_stream_start_time: Optional[float]` — monotonic timestamp for latency tracking
- `_session_id: str` — 8-char hex UUID for log correlation

### Event Handling Flow

```
handle_event(event) dispatches by event type:

Describe → write Info event back (service discovery)
Transcribe → store language preference
AudioStart → reset all per-stream state
AudioChunk → append to buffer, check early verify trigger
AudioStop → finalize (if not already responded)
```

**AudioChunk handler:**
1. If `_responded` is True, return immediately (consume silently)
2. Append chunk audio to `_audio_buffer`, capture rate/width/channels
3. If `_verify_started` is False, compute `buffered_seconds = len(buffer) / bytes_per_second`
4. If `buffered_seconds >= max_verify_seconds`:
   - Set `_verify_started = True`
   - Snapshot the buffer (`bytes(self._audio_buffer)`)
   - Create `asyncio.Task` for `_run_early_pipeline(snapshot)`

**AudioStop handler:**
1. If `_responded` is True, log and return
2. Otherwise call `_process_audio_sync()` (short audio fallback)

### _run_early_pipeline(verify_audio: bytes)

Called as a background task when 5s of audio is buffered.

1. Acquire `_MODEL_LOCK`
2. Run `verifier.verify(verify_audio, sample_rate)` in `run_in_executor` (thread pool, avoids blocking event loop)
3. Release lock
4. If rejected:
   - Cache result in `_verify_result_cache`
   - Return (wait for AudioStop to try full audio)
5. If matched:
   - Log speaker name, similarity, threshold
   - Trim audio with `_trim_for_asr()`
   - Forward to upstream ASR with `_forward_to_upstream()`
   - Write `Transcript` event back to HA
   - Set `_responded = True`
   - Log total pipeline time

### _process_audio_sync()

Called at AudioStop when early verify didn't respond (either short audio or early rejection).

1. If buffer is empty, return empty transcript
2. Check for `_verify_result_cache` (early rejection):
   - If cached: re-verify with full audio, keep best of cached vs full result
   - If no cache: first-time verification with full audio
3. Verification runs under `_MODEL_LOCK` in `run_in_executor`
4. If matched: trim, forward to ASR, write transcript
5. If rejected: log warning with all scores, write empty transcript

### _trim_for_asr(audio_bytes, result, bytes_per_second) → bytes

Simple first-N-seconds trim:
```python
max_bytes = int(self.asr_max_seconds * bytes_per_second)
return audio_bytes[:max_bytes]
```

The voice command is always at the start of the buffer (immediately after the wake word), so trimming from the end preserves the command while cutting trailing TV noise.

### _forward_to_upstream(audio_bytes) → str

Opens a new `AsyncClient` connection to the upstream ASR:
1. Send `Transcribe(language=self._language)` event
2. Send `AudioStart(rate, width, channels)` event
3. Stream audio in 100ms chunks (bytes_per_second // 10 bytes each)
4. Send `AudioStop` event
5. Read events until `Transcript` is received
6. Return transcript text
7. On any exception, log and return empty string

## Verifier (verify.py)

### Dataclass: VerificationResult

```python
@dataclass
class VerificationResult:
    is_match: bool                              # Whether any speaker exceeded threshold
    similarity: float                           # Best similarity score
    threshold: float                            # Threshold used
    matched_speaker: Optional[str] = None       # Name of matched speaker (None if rejected)
    all_scores: Dict[str, float] = field(...)   # {speaker_name: similarity} for all enrolled
    speech_audio: Optional[bytes] = None        # Extracted speech segment (repr=False)
    speech_start_sec: Optional[float] = None    # Speech segment start time
    speech_end_sec: Optional[float] = None      # Speech segment end time
```

### Class: SpeakerVerifier

**Constructor parameters:**
- `voiceprints_dir: str` — path to directory with .npy files
- `model_dir: str = "/data/models"` — HuggingFace model cache
- `device: str = "cuda"` — inference device
- `threshold: float = 0.20` — cosine similarity threshold
- `max_verify_seconds: float = 5.0` — first-pass audio length
- `window_seconds: float = 3.0` — sliding window size
- `step_seconds: float = 1.5` — sliding window step

**Initialization:**
1. Load ECAPA-TDNN via `EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=..., run_opts=...)`
   - `run_opts` is `{"device": "cuda"}` for GPU, empty dict `{}` for CPU
2. Load all `.npy` files from voiceprints_dir into `self.voiceprints: Dict[str, np.ndarray]`

### verify(audio_bytes, sample_rate) → VerificationResult

Three-pass verification strategy:

**Pass 1 — Speech segment:**
1. Call `_extract_speech_segment(audio_bytes, sample_rate)`
2. If speech found, verify the extracted segment
3. If match, return immediately

**Pass 2 — First N seconds:**
1. Take `audio_bytes[:max_verify_seconds * bytes_per_second]`
2. Verify this chunk
3. Track best result across passes
4. If match, return immediately

**Pass 3 — Sliding window:**
1. Only runs if audio is longer than max_verify_seconds AND at least one window_seconds long
2. Start offset at `step_bytes` (skip first window, already covered by Pass 2)
3. Slide a `window_seconds` window in `step_seconds` steps
4. Verify each window
5. Track best result
6. If any window matches, return immediately

If all passes fail, return the best result (which will have `is_match=False`).

### _extract_speech_segment(audio_bytes, sample_rate) → Optional[tuple[bytes, float, float]]

RMS energy-based speech detection:

1. Convert raw PCM bytes to int16 numpy array, cast to float32
2. Split into 50ms frames (`frame_samples = sample_rate * 0.05 = 800 samples`)
3. Compute RMS energy per frame: `sqrt(mean(frame ** 2))`
4. Find peak frame index (`argmax(rms)`)
5. If peak energy < 100 (near-silence), return None
6. Set energy threshold = peak_energy * 0.15
7. Expand outward from peak:
   - `start_frame`: decrement while `rms[start_frame - 1] >= threshold`
   - `end_frame`: increment while `rms[end_frame + 1] >= threshold`
8. Convert frame indices to byte offsets (frame_index * frame_samples * 2)
9. If segment < 1.0 seconds, expand symmetrically to reach minimum
10. Return `(segment_bytes, start_seconds, end_seconds)`

**Key parameters (hardcoded, not configurable):**
- Frame size: 50ms (800 samples at 16kHz)
- Energy threshold: 15% of peak
- Minimum segment: 1.0 seconds
- Near-silence cutoff: RMS < 100

### _verify_chunk(audio_bytes, sample_rate) → VerificationResult

1. Extract embedding via `_extract_embedding()`
2. Compute `1.0 - cosine(embedding, voiceprint)` for each enrolled speaker
3. Track best score and speaker
4. Return `VerificationResult` with `is_match = best_score >= threshold`

### _extract_embedding(audio_bytes, sample_rate) → np.ndarray

1. Convert raw PCM to int16 numpy array, cast to float32
2. Normalize to [-1.0, 1.0]: `audio_np /= 32768.0`
3. Wrap in torch tensor: `torch.tensor(audio_np).unsqueeze(0)` (adds batch dimension)
4. Move to CUDA if applicable
5. Run `classifier.encode_batch(signal)` under `torch.no_grad()`
6. Return squeezed numpy array (192 dimensions)

## Enrollment Script (scripts/enroll.py)

### Modes

**`--speaker <name>`**: Generate voiceprint
1. Create `data/enrollment/<name>/` if it doesn't exist (and return, prompting user to add WAV files)
2. Find all audio files (`.wav`, `.flac`, `.ogg`, `.mp3`)
3. Load ECAPA-TDNN model
4. For each audio file:
   - Load with `torchaudio.load()`
   - Resample to 16kHz if needed (`torchaudio.transforms.Resample`)
   - Convert to mono if needed (`signal.mean(dim=0, keepdim=True)`)
   - Skip if < 1.0 seconds
   - Extract embedding with `classifier.encode_batch(signal)`
5. Average all embeddings: `np.mean(embeddings, axis=0)`
6. L2-normalize: `voiceprint / np.linalg.norm(voiceprint)`
7. Save to `data/voiceprints/<name>.npy`

**`--list`**: List all enrolled speakers (glob `*.npy` in voiceprints dir)

**`--delete <name>`**: Delete `data/voiceprints/<name>.npy`

### Key Detail: Voiceprint Normalization

The enrolled voiceprint is L2-normalized (unit vector). The live embeddings from `_extract_embedding()` are NOT normalized. Cosine similarity is computed by `scipy.spatial.distance.cosine`, which handles normalization internally.

## Test Script (scripts/test_verify.py)

CLI tool for threshold tuning. Takes a WAV file, loads all voiceprints, computes similarity for each, and displays a formatted table with match/reject markers. Does not use the multi-pass strategy — just a single embedding comparison.

## Windows Recording Script (tools/record_samples.ps1)

PowerShell script that:
1. Detects audio devices via `ffmpeg -sources dshow` (with fallback to `-list_devices true`)
2. Lets user pick a microphone
3. Records N samples (default 30) of D seconds each (default 5)
4. Saves as 16kHz mono WAV in `data/enrollment/<speaker>/`
5. Shows suggested phrases from a list of 30 home automation commands
6. Filenames are timestamped: `<speaker>_YYYYMMDD_HHMMSS.wav`

## Wyoming Protocol Flow

The service communicates using the [Wyoming protocol](https://github.com/OHF-Voice/wyoming), which is a simple event-based TCP protocol. Events are JSON objects with a type and payload, terminated by newlines.

### Inbound events (from Home Assistant):

```
Describe         → Service discovery (responds with Info)
Transcribe       → Request with language preference
AudioStart       → Stream beginning (rate, width, channels)
AudioChunk       → Audio data (repeated, ~100ms per chunk)
AudioStop        → Stream end
```

### Outbound events (to Home Assistant):

```
Info             → Service capabilities (in response to Describe)
Transcript       → Transcription result (text string)
```

### Upstream ASR communication:

The proxy opens a NEW `AsyncClient` TCP connection to the upstream ASR for each verified audio stream. It replays the full Wyoming sequence (Transcribe → AudioStart → AudioChunks → AudioStop) and reads until it receives a Transcript event.

## Pipeline Timing

### Quiet room (short audio, VAD detects silence quickly):

```
0.0s  AudioStart
0.0-2.5s  AudioChunks arrive (voice command + brief silence)
2.5s  AudioStop (VAD detected silence)
      → _process_audio_sync() runs
      → Verify (~5-25ms GPU) → Forward to ASR → Transcript returned
~3.0s  Total
```

### Noisy room (TV keeps VAD open):

```
0.0s   AudioStart
0.0-5.0s  AudioChunks arrive (voice command + TV noise)
5.0s   Early verification triggers (_run_early_pipeline)
       → Verify (~5-25ms GPU) → Forward first 3s to ASR → Transcript returned
       → _responded = True
5.0-15.0s  More AudioChunks arrive (consumed silently)
15.0s  AudioStop (VAD timeout)
       → Already responded, log and return
~5.0s  Total (from user's perspective)
```

## Design Decisions

### Why early verification at 5 seconds (not sooner)?

Voice commands typically take 1-3 seconds. With 5 seconds of audio, we have the complete command plus enough context for reliable energy detection and speaker verification. Going shorter (e.g., 2-3s) risks verifying before the command is complete, reducing accuracy.

### Why simple first-N-seconds ASR trimming (not speech-bound trimming)?

We tried trimming audio to the speech segment bounds detected by energy analysis, but this caused issues:
- Trimming at arbitrary byte offsets mid-buffer confused Whisper (empty transcripts)
- The energy detector finds the peak, which may be mid-word, so the first word gets cut
- Adding padding to compensate made the logic fragile

The voice command is always at the start of the buffer (right after the wake word), so `audio[:max_bytes]` reliably captures it. Simple and robust.

### Why three verification passes?

Energy detection works well when the user's voice is significantly louder than background noise, but can fail when:
- The user speaks very softly (energy close to TV level)
- The TV has a loud moment during the command
- Unusual room acoustics

Pass 2 (first-N) and Pass 3 (sliding window) provide fallbacks for these edge cases without significantly increasing latency (each pass only runs if the previous one failed).

### Why respond before AudioStop?

The Wyoming protocol is request-response: HA sends the full audio stream, then reads the response. But the proxy can write the Transcript event to the TCP buffer at any time. When HA finishes sending and reads the socket, the transcript is already waiting. This doesn't violate the protocol — it just means the response is ready before HA asks for it.

The alternative (waiting for AudioStop) adds 10+ seconds of latency when TV noise keeps the VAD open. Users experience this as the satellite being stuck on "listening" for 15 seconds after they've finished speaking.

### Why asyncio.Lock instead of thread locks?

The ECAPA-TDNN model is not thread-safe for concurrent inference. We use `asyncio.Lock` because the handler is async, and we run inference in `run_in_executor` (thread pool) while holding the lock. This ensures only one inference runs at a time while allowing other async operations (buffering, protocol I/O) to proceed concurrently.

### Why cosine similarity (not Euclidean distance)?

Cosine similarity measures the angle between embedding vectors, making it invariant to magnitude differences. This is important because different audio durations and volumes produce embeddings with different magnitudes but similar directions. The ECAPA-TDNN VoxCeleb benchmark uses cosine similarity as the standard metric.

## Typical Similarity Scores

These are approximate ranges observed during testing:

| Source | Score Range |
|--------|------------|
| Enrolled speaker (clear, close) | 0.45–0.75 |
| Enrolled speaker (quiet, far) | 0.20–0.45 |
| TV dialogue | 0.05–0.20 |
| Different person | 0.05–0.25 |
| Near-silence / noise only | -0.05–0.10 |

Default threshold of 0.20 is deliberately low to minimize false rejections in noisy environments. Users in quiet environments can increase to 0.30–0.45 for tighter security.

## Docker Hub

Images are published as:
- `jxlarrea/wyoming-voice-match:latest` — GPU image
- `jxlarrea/wyoming-voice-match:cpu` — CPU image

GitHub repository: `https://github.com/jxlarrea/wyoming-voice-match`

## Known Quirks & Gotchas

### _verify_result_cache is not declared in __init__

The `_verify_result_cache` attribute is set dynamically in `_run_early_pipeline()` and read with `getattr(self, '_verify_result_cache', None)` in `_process_audio_sync()`. This works because the cache is only read at AudioStop, which always comes after AudioChunk processing. It was done this way to avoid adding another Optional field to __init__ — a future refactor could declare it properly.

### Wyoming library API surface used

The project uses these specific classes from the `wyoming` package:
- `wyoming.server.AsyncServer` — TCP server, created via `AsyncServer.from_uri(uri)`
- `wyoming.server.AsyncEventHandler` — base class for handlers, provides `write_event()` and `handle_event()` interface
- `wyoming.client.AsyncClient` — TCP client for upstream ASR, used as async context manager via `AsyncClient.from_uri(uri)`
- `wyoming.event.Event` — base event class with `.type` attribute
- `wyoming.asr.Transcribe` — request event with `.language` attribute
- `wyoming.asr.Transcript` — response event with `.text` attribute
- `wyoming.audio.AudioStart` — stream start with `.rate`, `.width`, `.channels`
- `wyoming.audio.AudioChunk` — audio data with `.audio` (bytes), `.rate`, `.width`, `.channels`
- `wyoming.audio.AudioStop` — stream end (no payload)
- `wyoming.info.Describe` — service discovery request
- `wyoming.info.Info` — service capabilities, contains list of `AsrProgram`
- `wyoming.info.AsrProgram` — program metadata with `name`, `description`, `attribution`, `models`
- `wyoming.info.AsrModel` — model metadata with `name`, `description`, `languages`, `attribution`
- `wyoming.info.Attribution` — `name` and `url`

All event types have `.is_type(event_type_string)` class method and `.from_event(event)` factory. Events are created via `SomeEvent(...).event()`.

The `AsyncServer.run()` method takes a handler factory (callable that returns a handler instance). We use `functools.partial(SpeakerVerifyHandler, wyoming_info, verifier, upstream_uri, asr_max_seconds)`. The server passes `*args, **kwargs` to the factory, which the handler forwards to `super().__init__()`.

### SpeechBrain API surface used

- `speechbrain.inference.speaker.EncoderClassifier.from_hparams(source=..., savedir=..., run_opts=...)` — loads pretrained model
- `classifier.encode_batch(signal)` — takes a `torch.Tensor` of shape `(1, num_samples)`, returns embedding tensor

The model downloads from HuggingFace Hub on first use and caches in `savedir`. The `run_opts={"device": "cuda"}` places the model on GPU. For CPU, pass an empty dict `{}` (NOT `{"device": "cpu"}` — SpeechBrain handles CPU as default).

### Audio format assumptions

- The service assumes 16-bit signed little-endian PCM throughout
- `bytes_per_second = sample_rate * 2` (hardcoded assumption of 16-bit mono)
- The handler tracks `_audio_width` and `_audio_channels` but the verifier only uses `sample_rate`
- Enrollment script uses `torchaudio.load()` which handles any audio format, then resamples to 16kHz mono

### CUDA fallback in enrollment

The enrollment script checks `torch.cuda.is_available()` and falls back to CPU if CUDA is unavailable. The main service does NOT do this — if `DEVICE=cuda` but CUDA is unavailable, it will crash at startup. This is intentional: in production, you want to know immediately if GPU acceleration is missing.

## Approaches Tried and Abandoned

These were attempted during development but didn't work well. Documenting them to prevent a future rebuild from repeating the same mistakes.

### Speech-bound ASR trimming

**What:** Trim the audio sent to ASR using the speech segment bounds from energy detection (speech_start - padding to speech_end + ASR_MAX_SECONDS).

**Why it failed:** Trimming mid-buffer at arbitrary byte offsets produced audio that confused Whisper, resulting in empty transcripts. The energy detector also finds the peak energy frame, which may be in the middle of a word — not the start of speech. Adding padding before the detected start helped but was fragile. The first word was still frequently truncated (e.g., "the current president" instead of "Who is the current president") because the quiet onset of speech falls below the 15% energy threshold.

**What works instead:** Simple `audio[:max_bytes]` from buffer start. The voice command is always at the start (right after wake word), so this reliably captures it.

### Waiting for AudioStop before responding

**What:** Buffer all audio, verify at AudioStop, then forward to ASR.

**Why it failed:** When a TV is playing, Home Assistant's VAD can't detect silence, so the audio stream stays open for 15+ seconds (the VAD timeout). The user waits 15+ seconds for a response to a 2-second command.

**What works instead:** Start verification at 5 seconds, respond immediately when verified, consume remaining chunks silently.

### Lower energy threshold for speech detection

**What:** Tried 5% and 10% of peak energy instead of 15%.

**Why it failed:** Too much background noise included in the speech segment, reducing verification accuracy. The TV's audio was being included alongside the voice command.

**What works instead:** 15% of peak provides a good balance — captures the full voice command while excluding most background noise. The three-pass strategy handles cases where energy detection misses the voice.

### Higher default verification threshold

**What:** Started with 0.45, then 0.30.

**Why it failed:** Users were being rejected too frequently, especially when speaking softly or from a distance. In noisy environments with TV audio, similarity scores drop because the speech segment still contains some background noise.

**What works instead:** 0.20 default with guidance to increase if needed. TV audio typically scores 0.05–0.20, so 0.20 still provides good separation while accepting quiet commands.

## Appendix: Complete Source Files

Every file in the project, in full. These are the canonical versions — if the prose above contradicts the code below, the code is correct.

### wyoming_voice_match/__init__.py

```python
"""Wyoming Voice Match — ASR proxy with speaker verification."""

__version__ = "1.0.0"
```

### wyoming_voice_match/__main__.py

```python
"""Entry point for wyoming-voice-match."""

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import SpeakerVerifyHandler
from .verify import SpeakerVerifier

_LOGGER = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        description="Wyoming ASR proxy with speaker verification"
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("LISTEN_URI", "tcp://0.0.0.0:10350"),
        help="URI to listen on (default: tcp://0.0.0.0:10350)",
    )
    parser.add_argument(
        "--upstream-uri",
        default=os.environ.get("UPSTREAM_URI", "tcp://localhost:10300"),
        help="URI of upstream ASR service (default: tcp://localhost:10300)",
    )
    parser.add_argument(
        "--voiceprints-dir",
        default=os.environ.get("VOICEPRINTS_DIR", "/data/voiceprints"),
        help="Directory containing voiceprint .npy files (default: /data/voiceprints)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("VERIFY_THRESHOLD", "0.20")),
        help="Cosine similarity threshold for verification (default: 0.20)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/data/models"),
        help="Directory to cache the speaker model (default: /data/models)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--max-verify-seconds",
        type=float,
        default=float(os.environ.get("MAX_VERIFY_SECONDS", "5.0")),
        help="Max audio duration (seconds) for first-pass verification (default: 5.0)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=float(os.environ.get("VERIFY_WINDOW_SECONDS", "3.0")),
        help="Sliding window size in seconds for fallback verification (default: 3.0)",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=float(os.environ.get("VERIFY_STEP_SECONDS", "1.5")),
        help="Sliding window step in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--asr-max-seconds",
        type=float,
        default=float(os.environ.get("ASR_MAX_SECONDS", "3.0")),
        help="Max audio duration (seconds) forwarded to upstream ASR (default: 3.0)",
    )

    return parser.parse_args()


async def main() -> None:
    """Run the Wyoming voice match proxy."""
    args = get_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate voiceprints directory exists
    voiceprints_dir = Path(args.voiceprints_dir)
    if not voiceprints_dir.exists():
        _LOGGER.error(
            "Voiceprints directory not found at %s. "
            "Run the enrollment script first: "
            "python -m scripts.enroll --speaker <name>",
            voiceprints_dir,
        )
        sys.exit(1)

    # Load speaker verifier
    _LOGGER.info("Loading ECAPA-TDNN speaker verification model...")
    verifier = SpeakerVerifier(
        voiceprints_dir=str(voiceprints_dir),
        model_dir=args.model_dir,
        device=args.device,
        threshold=args.threshold,
        max_verify_seconds=args.max_verify_seconds,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    if not verifier.voiceprints:
        _LOGGER.error(
            "No voiceprints found in %s. "
            "Run the enrollment script first: "
            "python -m scripts.enroll --speaker <name>",
            voiceprints_dir,
        )
        sys.exit(1)

    _LOGGER.info(
        "Speaker verifier ready — %d speaker(s) enrolled "
        "(threshold=%.2f, device=%s, verify_window=%.1fs, "
        "sliding_window=%.1fs/%.1fs)",
        len(verifier.voiceprints),
        args.threshold,
        args.device,
        args.max_verify_seconds,
        args.window_seconds,
        args.step_seconds,
    )

    # Build Wyoming service info
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="voice-match",
                description=f"Speaker-verified ASR proxy v{__version__}",
                attribution=Attribution(
                    name="Wyoming Voice Match",
                    url="https://github.com/jxlarrea/wyoming-voice-match",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="voice-match-proxy",
                        description="ECAPA-TDNN speaker gate → upstream ASR",
                        languages=["en"],
                        attribution=Attribution(
                            name="Wyoming Voice Match",
                            url="https://github.com/jxlarrea/wyoming-voice-match",
                        ),
                        installed=True,
                        version=__version__,
                    )
                ],
            )
        ]
    )

    _LOGGER.info(
        "Starting server on %s → upstream %s",
        args.uri,
        args.upstream_uri,
    )

    server = AsyncServer.from_uri(args.uri)
    await server.run(
        partial(
            SpeakerVerifyHandler,
            wyoming_info,
            verifier,
            args.upstream_uri,
            args.asr_max_seconds,
        )
    )


def run() -> None:
    """Sync wrapper for main."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
```

### wyoming_voice_match/handler.py

```python
"""Wyoming event handler for speaker-verified ASR proxy."""

import asyncio
import logging
import time
import uuid
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .verify import SpeakerVerifier, VerificationResult

_LOGGER = logging.getLogger(__name__)

# Lock to prevent concurrent model inference
_MODEL_LOCK = asyncio.Lock()

# Extra audio before detected speech start to capture quiet lead-in syllables
_ASR_PADDING_SEC = 0.5


class SpeakerVerifyHandler(AsyncEventHandler):
    """Wyoming ASR handler that gates transcription on speaker identity.

    Runs speaker verification early (as soon as enough audio is buffered)
    and immediately forwards to ASR without waiting for AudioStop. This
    bypasses the upstream VAD latency when background noise keeps the
    stream open.
    """

    def __init__(
        self,
        wyoming_info: Info,
        verifier: SpeakerVerifier,
        upstream_uri: str,
        asr_max_seconds: float = 3.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info = wyoming_info
        self.verifier = verifier
        self.upstream_uri = upstream_uri
        self.asr_max_seconds = asr_max_seconds

        # Per-connection state
        self._audio_buffer = bytes()
        self._audio_rate: int = 16000
        self._audio_width: int = 2
        self._audio_channels: int = 1
        self._language: Optional[str] = None
        self._verify_task: Optional[asyncio.Task] = None
        self._verify_started: bool = False
        self._responded: bool = False
        self._stream_start_time: Optional[float] = None
        self._session_id: str = uuid.uuid4().hex[:8]

    async def handle_event(self, event: Event) -> bool:
        """Process a single Wyoming event."""
        sid = self._session_id

        # Service discovery
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        # Transcription request — capture language preference
        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language
            return True

        # Audio stream start — reset state
        if AudioStart.is_type(event.type):
            self._audio_buffer = bytes()
            self._verify_task = None
            self._verify_started = False
            self._responded = False
            self._stream_start_time = time.monotonic()
            _LOGGER.debug("[%s] ── New audio session started ──", sid)
            return True

        # Audio data — accumulate and trigger early verification + ASR
        if AudioChunk.is_type(event.type):
            # If we already responded, just consume remaining chunks
            if self._responded:
                return True

            chunk = AudioChunk.from_event(event)
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._audio_buffer += chunk.audio

            # Trigger verification once we have enough audio
            if not self._verify_started:
                bytes_per_second = (
                    self._audio_rate * self._audio_width * self._audio_channels
                )
                buffered_seconds = len(self._audio_buffer) / bytes_per_second

                if buffered_seconds >= self.verifier.max_verify_seconds:
                    self._verify_started = True
                    _LOGGER.debug(
                        "[%s] Early verify: %.1fs buffered, starting verification",
                        sid, buffered_seconds,
                    )
                    # Take a snapshot of the buffer for verification
                    verify_audio = bytes(self._audio_buffer)
                    self._verify_task = asyncio.create_task(
                        self._run_early_pipeline(verify_audio)
                    )

            return True

        # Audio stream end — respond if we haven't already
        if AudioStop.is_type(event.type):
            if self._responded:
                # Already sent response during streaming
                elapsed = self._elapsed_ms()
                _LOGGER.debug(
                    "[%s] AudioStop received (already responded, %.0fms since start)",
                    sid, elapsed,
                )
                return True

            # Short audio — never triggered early verification
            await self._process_audio_sync()
            return True

        return True

    async def _run_early_pipeline(self, verify_audio: bytes) -> None:
        """Run verification and, if matched, immediately forward to ASR."""
        sid = self._session_id

        # Run speaker verification
        async with _MODEL_LOCK:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self.verifier.verify,
                verify_audio,
                self._audio_rate,
            )

        if not result.is_match:
            # Don't respond yet — wait for AudioStop in case more audio
            # changes the outcome (handled in _process_audio_sync)
            _LOGGER.debug(
                "[%s] Early verify rejected (%.4f), waiting for AudioStop",
                sid, result.similarity,
            )
            self._verify_result_cache = result
            return

        _LOGGER.info(
            "[%s] Speaker verified: %s (similarity=%.4f, threshold=%.2f), "
            "forwarding to ASR immediately",
            sid, result.matched_speaker, result.similarity, result.threshold,
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            for name, score in result.all_scores.items():
                _LOGGER.debug("[%s]   %s: %.4f", sid, name, score)

        # Trim audio for ASR: from speech_start (with padding) to
        # speech_end + asr_max_seconds (to allow for longer commands
        # that extend beyond the detected energy peak)
        bytes_per_second = (
            self._audio_rate * self._audio_width * self._audio_channels
        )
        asr_audio = self._trim_for_asr(verify_audio, result, bytes_per_second)
        audio_duration = len(verify_audio) / bytes_per_second
        asr_duration = len(asr_audio) / bytes_per_second
        _LOGGER.debug(
            "[%s] Forwarding %.1fs (trimmed from %.1fs) to ASR",
            sid, asr_duration, audio_duration,
        )

        # Forward to ASR and respond immediately
        transcript = await self._forward_to_upstream(asr_audio)
        await self.write_event(Transcript(text=transcript).event())
        self._responded = True

        total_elapsed = self._elapsed_ms()
        _LOGGER.info(
            "[%s] Pipeline complete in %.0fms: \"%s\"",
            sid, total_elapsed, transcript,
        )

    async def _process_audio_sync(self) -> None:
        """Fallback: verify and forward when AudioStop arrives (short audio)."""
        sid = self._session_id
        audio_bytes = self._audio_buffer
        bytes_per_second = (
            self._audio_rate * self._audio_width * self._audio_channels
        )
        audio_duration = len(audio_bytes) / bytes_per_second

        if len(audio_bytes) == 0:
            _LOGGER.debug("[%s] Empty audio buffer, returning empty transcript", sid)
            await self.write_event(Transcript(text="").event())
            return

        stream_elapsed = self._elapsed_ms()
        _LOGGER.debug(
            "[%s] AudioStop received: %.1fs of audio (%d bytes), "
            "stream duration: %.0fms",
            sid, audio_duration, len(audio_bytes), stream_elapsed,
        )

        # Check if early verification ran but was rejected
        cached = getattr(self, '_verify_result_cache', None)
        if cached is not None:
            # Early verify rejected — try full audio now
            _LOGGER.debug(
                "[%s] Re-verifying with full %.1fs audio",
                sid, audio_duration,
            )
            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self.verifier.verify,
                    audio_bytes,
                    self._audio_rate,
                )
            # Use best of early and full
            if cached.similarity > result.similarity:
                result = cached
        else:
            # No early verification was triggered — verify now
            _LOGGER.debug(
                "[%s] No early verification (only %.1fs buffered), verifying now",
                sid, audio_duration,
            )
            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self.verifier.verify,
                    audio_bytes,
                    self._audio_rate,
                )

        if result.is_match:
            _LOGGER.info(
                "[%s] Speaker verified: %s (similarity=%.4f, threshold=%.2f), "
                "forwarding to ASR",
                sid, result.matched_speaker, result.similarity, result.threshold,
            )
            if _LOGGER.isEnabledFor(logging.DEBUG):
                for name, score in result.all_scores.items():
                    _LOGGER.debug("[%s]   %s: %.4f", sid, name, score)

            # Trim audio for ASR
            asr_audio = self._trim_for_asr(audio_bytes, result, bytes_per_second)
            asr_duration = len(asr_audio) / bytes_per_second
            _LOGGER.debug(
                "[%s] Forwarding %.1fs (trimmed from %.1fs) to ASR",
                sid, asr_duration, audio_duration,
            )

            transcript = await self._forward_to_upstream(asr_audio)
            await self.write_event(Transcript(text=transcript).event())
            self._responded = True
            total_elapsed = self._elapsed_ms()
            _LOGGER.info(
                "[%s] Pipeline complete in %.0fms: \"%s\"",
                sid, total_elapsed, transcript,
            )
        else:
            total_elapsed = self._elapsed_ms()
            _LOGGER.warning(
                "[%s] Speaker rejected in %.0fms (best=%.4f, threshold=%.2f, scores=%s)",
                sid, total_elapsed, result.similarity, result.threshold,
                {n: f"{s:.4f}" for n, s in result.all_scores.items()},
            )
            await self.write_event(Transcript(text="").event())
            self._responded = True

    def _trim_for_asr(
        self,
        audio_bytes: bytes,
        result: VerificationResult,
        bytes_per_second: int,
    ) -> bytes:
        """Trim audio for ASR.

        Sends the first asr_max_seconds of audio from the buffer.
        The voice command is always at the start of the buffer (right
        after the wake word), so this captures it while cutting off
        trailing background noise from the VAD keeping the stream open.
        """
        max_bytes = int(self.asr_max_seconds * bytes_per_second)
        trimmed = audio_bytes[:max_bytes]
        _LOGGER.debug(
            "[%s] ASR trim: first %.1fs of %.1fs",
            self._session_id,
            len(trimmed) / bytes_per_second,
            len(audio_bytes) / bytes_per_second,
        )
        return trimmed

    def _elapsed_ms(self) -> float:
        """Milliseconds since stream start."""
        if self._stream_start_time is not None:
            return (time.monotonic() - self._stream_start_time) * 1000
        return 0.0

    async def _forward_to_upstream(self, audio_bytes: bytes) -> str:
        """Forward verified audio to the upstream ASR service."""
        try:
            async with AsyncClient.from_uri(self.upstream_uri) as client:
                # Send transcription request
                await client.write_event(
                    Transcribe(language=self._language).event()
                )

                # Send audio start
                await client.write_event(
                    AudioStart(
                        rate=self._audio_rate,
                        width=self._audio_width,
                        channels=self._audio_channels,
                    ).event()
                )

                # Stream audio in chunks (100ms per chunk)
                bytes_per_chunk = (
                    self._audio_rate * self._audio_width * self._audio_channels
                ) // 10
                for offset in range(0, len(audio_bytes), bytes_per_chunk):
                    chunk_data = audio_bytes[offset : offset + bytes_per_chunk]
                    await client.write_event(
                        AudioChunk(
                            audio=chunk_data,
                            rate=self._audio_rate,
                            width=self._audio_width,
                            channels=self._audio_channels,
                        ).event()
                    )

                # Signal end of audio
                await client.write_event(AudioStop().event())

                # Wait for transcript response
                while True:
                    response = await client.read_event()
                    if response is None:
                        _LOGGER.error("Upstream ASR closed connection unexpectedly")
                        return ""
                    if Transcript.is_type(response.type):
                        transcript = Transcript.from_event(response)
                        _LOGGER.debug("[%s] Upstream transcript: %s", self._session_id, transcript.text)
                        return transcript.text

        except Exception:
            _LOGGER.exception("Error communicating with upstream ASR at %s", self.upstream_uri)
            return ""
```

### wyoming_voice_match/verify.py

```python
"""Speaker verification using SpeechBrain ECAPA-TDNN."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier

_LOGGER = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a speaker verification check."""

    is_match: bool
    similarity: float
    threshold: float
    matched_speaker: Optional[str] = None
    all_scores: Dict[str, float] = field(default_factory=dict)
    speech_audio: Optional[bytes] = field(default=None, repr=False)
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None


class SpeakerVerifier:
    """Verifies speaker identity against one or more enrolled voiceprints.

    Uses SpeechBrain's pretrained ECAPA-TDNN model to extract 192-dimensional
    speaker embeddings and compares them via cosine similarity.

    Supports multiple speakers — audio is accepted if any enrolled voice
    matches above the threshold.
    """

    def __init__(
        self,
        voiceprints_dir: str,
        model_dir: str = "/data/models",
        device: str = "cuda",
        threshold: float = 0.20,
        max_verify_seconds: float = 5.0,
        window_seconds: float = 3.0,
        step_seconds: float = 1.5,
    ) -> None:
        self.threshold = threshold
        self.device = device
        self.max_verify_seconds = max_verify_seconds
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds

        # Load the pretrained ECAPA-TDNN model
        run_opts = {"device": device} if device == "cuda" else {}
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=f"{model_dir}/spkrec-ecapa-voxceleb",
            run_opts=run_opts,
        )

        # Load all enrolled voiceprints
        self.voiceprints: Dict[str, np.ndarray] = {}
        self._load_voiceprints(voiceprints_dir)

    def _load_voiceprints(self, voiceprints_dir: str) -> None:
        """Load all .npy voiceprint files from the directory."""
        vp_path = Path(voiceprints_dir)
        if not vp_path.exists():
            _LOGGER.warning("Voiceprints directory not found: %s", vp_path)
            return

        for npy_file in sorted(vp_path.glob("*.npy")):
            speaker_name = npy_file.stem
            voiceprint = np.load(str(npy_file))
            self.voiceprints[speaker_name] = voiceprint
            _LOGGER.info(
                "Loaded voiceprint: %s (shape=%s)",
                speaker_name,
                voiceprint.shape,
            )

        if not self.voiceprints:
            _LOGGER.warning(
                "No voiceprints found in %s. "
                "Run the enrollment script first.",
                vp_path,
            )

    def reload_voiceprints(self, voiceprints_dir: str) -> None:
        """Reload voiceprints from disk (e.g., after re-enrollment)."""
        self.voiceprints.clear()
        self._load_voiceprints(voiceprints_dir)

    def verify(self, audio_bytes: bytes, sample_rate: int = 16000) -> VerificationResult:
        """Verify if audio matches any enrolled speaker.

        Uses a multi-pass strategy to handle background noise:
        1. Speech pass: extract the highest-energy segment (likely the voice
           command) and verify just that.
        2. First-N pass: verify only the first MAX_VERIFY_SECONDS of audio.
        3. Sliding window pass: scan the full audio with overlapping windows.

        Full audio is still forwarded to ASR regardless of which pass matched.

        Args:
            audio_bytes: Raw PCM audio (16-bit signed little-endian).
            sample_rate: Audio sample rate in Hz.

        Returns:
            VerificationResult with match status, best similarity score,
            matched speaker name, and scores for all enrolled speakers.
        """
        if not self.voiceprints:
            _LOGGER.warning("No voiceprints enrolled — rejecting audio")
            return VerificationResult(
                is_match=False,
                similarity=0.0,
                threshold=self.threshold,
            )

        start_time = time.monotonic()
        bytes_per_second = sample_rate * 2  # 16-bit = 2 bytes per sample
        audio_duration = len(audio_bytes) / bytes_per_second
        best_result: Optional[VerificationResult] = None
        speech_chunk: Optional[bytes] = None
        speech_start_sec: Optional[float] = None
        speech_end_sec: Optional[float] = None

        # --- Pass 1: energy-based speech segment ---
        pass1_start = time.monotonic()
        speech_result = self._extract_speech_segment(audio_bytes, sample_rate)
        if speech_result is not None:
            speech_chunk, speech_start_sec, speech_end_sec = speech_result
            speech_duration = len(speech_chunk) / bytes_per_second
            _LOGGER.debug(
                "Pass 1 (speech): verifying %.1fs speech segment from %.1fs audio",
                speech_duration,
                audio_duration,
            )
            result = self._verify_chunk(speech_chunk, sample_rate)
            best_result = result
            pass1_elapsed = (time.monotonic() - pass1_start) * 1000

            if result.is_match:
                _LOGGER.debug(
                    "Pass 1 (speech) matched in %.0fms (%.4f)",
                    pass1_elapsed, result.similarity,
                )
                _LOGGER.debug("Total verification time: %.0fms", pass1_elapsed)
                result.speech_audio = speech_chunk
                result.speech_start_sec = speech_start_sec
                result.speech_end_sec = speech_end_sec
                return result

            _LOGGER.debug(
                "Pass 1 (speech) rejected in %.0fms (%.4f)",
                pass1_elapsed, result.similarity,
            )
        else:
            pass1_elapsed = (time.monotonic() - pass1_start) * 1000
            _LOGGER.debug(
                "Pass 1 (speech) skipped — no speech segment detected (%.0fms)",
                pass1_elapsed,
            )

        # --- Pass 2: first N seconds ---
        pass2_start = time.monotonic()
        max_bytes = int(self.max_verify_seconds * bytes_per_second)
        first_chunk = audio_bytes[:max_bytes]

        _LOGGER.debug(
            "Pass 2 (first-N): verifying first %.1fs",
            len(first_chunk) / bytes_per_second,
        )

        result = self._verify_chunk(first_chunk, sample_rate)
        if best_result is None or result.similarity > best_result.similarity:
            best_result = result
        pass2_elapsed = (time.monotonic() - pass2_start) * 1000

        if result.is_match:
            total = (time.monotonic() - start_time) * 1000
            _LOGGER.debug(
                "Pass 2 (first-N) matched in %.0fms (%.4f)",
                pass2_elapsed, result.similarity,
            )
            _LOGGER.debug("Total verification time: %.0fms", total)
            result.speech_audio = speech_chunk
            result.speech_start_sec = speech_start_sec
            result.speech_end_sec = speech_end_sec
            return result

        _LOGGER.debug(
            "Pass 2 (first-N) rejected in %.0fms (%.4f)",
            pass2_elapsed, result.similarity,
        )

        # --- Pass 3: sliding window over full audio ---
        window_bytes = int(self.window_seconds * bytes_per_second)
        step_bytes = int(self.step_seconds * bytes_per_second)

        if len(audio_bytes) > max_bytes and len(audio_bytes) >= window_bytes:
            pass3_start = time.monotonic()
            _LOGGER.debug(
                "Pass 3 (sliding): %.1fs window with %.1fs step over %.1fs audio",
                self.window_seconds,
                self.step_seconds,
                audio_duration,
            )

            offset = step_bytes  # skip first window (covered by pass 2)
            window_count = 0

            while offset + window_bytes <= len(audio_bytes):
                window = audio_bytes[offset : offset + window_bytes]
                window_result = self._verify_chunk(window, sample_rate)
                window_count += 1

                if window_result.similarity > best_result.similarity:
                    best_result = window_result

                window_start = offset / bytes_per_second
                _LOGGER.debug(
                    "  Window %d (%.1f-%.1fs): %.4f",
                    window_count,
                    window_start,
                    window_start + self.window_seconds,
                    window_result.similarity,
                )

                if window_result.is_match:
                    pass3_elapsed = (time.monotonic() - pass3_start) * 1000
                    total = (time.monotonic() - start_time) * 1000
                    _LOGGER.debug(
                        "Pass 3 (sliding) matched window %d in %.0fms (%.4f)",
                        window_count, pass3_elapsed, window_result.similarity,
                    )
                    _LOGGER.debug("Total verification time: %.0fms", total)
                    window_result.speech_audio = speech_chunk
                    window_result.speech_start_sec = speech_start_sec
                    window_result.speech_end_sec = speech_end_sec
                    return window_result

                offset += step_bytes

            pass3_elapsed = (time.monotonic() - pass3_start) * 1000
            _LOGGER.debug(
                "Pass 3 (sliding) rejected after %d windows in %.0fms (best=%.4f)",
                window_count, pass3_elapsed, best_result.similarity,
            )

        total = (time.monotonic() - start_time) * 1000
        _LOGGER.debug(
            "All passes rejected — total: %.0fms (best=%.4f)",
            total, best_result.similarity,
        )
        return best_result

    def _extract_speech_segment(
        self, audio_bytes: bytes, sample_rate: int
    ) -> Optional[tuple]:
        """Extract the segment of audio with the highest energy (likely speech).

        Computes RMS energy in short frames, finds the peak region, and
        expands outward until energy drops below a fraction of the peak.
        Returns (speech_bytes, start_seconds) or None if audio is too short.
        """
        bytes_per_second = sample_rate * 2
        min_segment_bytes = int(1.0 * bytes_per_second)  # at least 1 second

        if len(audio_bytes) < min_segment_bytes:
            return None

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # Compute RMS energy in 50ms frames
        frame_samples = int(sample_rate * 0.05)
        num_frames = len(audio_np) // frame_samples

        if num_frames < 2:
            return None

        frames = audio_np[: num_frames * frame_samples].reshape(num_frames, frame_samples)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Find the frame with peak energy
        peak_idx = int(np.argmax(rms))
        peak_energy = rms[peak_idx]

        if peak_energy < 100:  # near-silence, skip
            return None

        # Expand outward from peak while energy stays above 15% of peak
        energy_threshold = peak_energy * 0.15

        start_frame = peak_idx
        while start_frame > 0 and rms[start_frame - 1] >= energy_threshold:
            start_frame -= 1

        end_frame = peak_idx
        while end_frame < num_frames - 1 and rms[end_frame + 1] >= energy_threshold:
            end_frame += 1

        # Convert frame indices back to byte offsets
        start_byte = start_frame * frame_samples * 2
        end_byte = (end_frame + 1) * frame_samples * 2

        segment = audio_bytes[start_byte:end_byte]

        # Ensure minimum length
        if len(segment) < min_segment_bytes:
            # Expand symmetrically to reach minimum
            deficit = min_segment_bytes - len(segment)
            expand = deficit // 2
            start_byte = max(0, start_byte - expand)
            end_byte = min(len(audio_bytes), end_byte + expand)
            segment = audio_bytes[start_byte:end_byte]

        segment_duration = len(segment) / bytes_per_second
        offset_seconds = start_byte / bytes_per_second
        _LOGGER.debug(
            "Speech detected: %.1f-%.1fs (%.1fs segment, peak_energy=%.0f)",
            offset_seconds,
            offset_seconds + segment_duration,
            segment_duration,
            peak_energy,
        )

        return segment, offset_seconds, offset_seconds + segment_duration

    def _verify_chunk(self, audio_bytes: bytes, sample_rate: int) -> VerificationResult:
        """Verify a single chunk of audio against all enrolled voiceprints."""
        embedding = self._extract_embedding(audio_bytes, sample_rate)

        all_scores: Dict[str, float] = {}
        best_score = -1.0
        best_speaker: Optional[str] = None

        for speaker_name, voiceprint in self.voiceprints.items():
            similarity = 1.0 - cosine(embedding, voiceprint)
            all_scores[speaker_name] = float(similarity)

            if similarity > best_score:
                best_score = similarity
                best_speaker = speaker_name

        is_match = best_score >= self.threshold

        return VerificationResult(
            is_match=is_match,
            similarity=float(best_score),
            threshold=self.threshold,
            matched_speaker=best_speaker if is_match else None,
            all_scores=all_scores,
        )

    def extract_embedding(self, audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Extract a speaker embedding from audio. Public API for enrollment."""
        return self._extract_embedding(audio_bytes, sample_rate)

    def _extract_embedding(self, audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Extract a 192-dimensional speaker embedding from raw PCM audio."""
        # Convert raw PCM bytes to float tensor
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np /= 32768.0  # Normalize to [-1.0, 1.0]

        signal = torch.tensor(audio_np).unsqueeze(0)

        if self.device == "cuda":
            signal = signal.to("cuda")

        with torch.no_grad():
            embedding = self.classifier.encode_batch(signal)

        return embedding.squeeze().cpu().numpy()
```

### scripts/__init__.py

```python
"""Utility scripts for Wyoming Voice Match."""
```

### scripts/enroll.py

```python
"""Enrollment script — generate a voiceprint from WAV samples.

Usage:
    python -m scripts.enroll --speaker juan [--enrollment-dir /data/enrollment]
    python -m scripts.enroll --speaker maria
    python -m scripts.enroll --list

Place 16kHz mono WAV files in data/enrollment/<speaker_name>/, then run this
script to compute an averaged speaker embedding (voiceprint).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def main() -> None:
    """Run enrollment to generate a voiceprint."""
    parser = argparse.ArgumentParser(
        description="Enroll speaker voice samples to create a voiceprint"
    )
    parser.add_argument(
        "--speaker",
        help="Name of the speaker to enroll (creates data/enrollment/<name>/ and data/voiceprints/<name>.npy)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all enrolled speakers",
    )
    parser.add_argument(
        "--delete",
        metavar="NAME",
        help="Delete an enrolled speaker's voiceprint",
    )
    parser.add_argument(
        "--enrollment-dir",
        default=os.environ.get("ENROLLMENT_DIR", "/data/enrollment"),
        help="Root directory containing speaker subdirectories with WAV files",
    )
    parser.add_argument(
        "--voiceprints-dir",
        default=os.environ.get("VOICEPRINTS_DIR", "/data/voiceprints"),
        help="Output directory for voiceprint .npy files",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/data/models"),
        help="Directory to cache the speaker model",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    args = parser.parse_args()

    voiceprints_dir = Path(args.voiceprints_dir)
    enrollment_dir = Path(args.enrollment_dir)

    # Handle --list
    if args.list:
        voiceprints_dir.mkdir(parents=True, exist_ok=True)
        speakers = sorted(f.stem for f in voiceprints_dir.glob("*.npy"))
        if speakers:
            print(f"\nEnrolled speakers ({len(speakers)}):")
            for name in speakers:
                print(f"  • {name}")
            print()
        else:
            print("\nNo speakers enrolled yet.")
            print(f"Run: python -m scripts.enroll --speaker <name>\n")
        return

    # Handle --delete
    if args.delete:
        vp_file = voiceprints_dir / f"{args.delete}.npy"
        if vp_file.exists():
            vp_file.unlink()
            _LOGGER.info("Deleted voiceprint for '%s'", args.delete)
        else:
            _LOGGER.error("No voiceprint found for '%s'", args.delete)
            sys.exit(1)
        return

    # Handle --speaker (enrollment)
    if not args.speaker:
        parser.print_help()
        print("\nError: --speaker, --list, or --delete is required.")
        sys.exit(1)

    speaker_name = args.speaker.strip().lower()
    speaker_dir = enrollment_dir / speaker_name

    if not speaker_dir.exists():
        speaker_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info(
            "Created enrollment directory: %s", speaker_dir
        )
        _LOGGER.info(
            "Place WAV files (16kHz, mono, 3-10s each) in this directory, "
            "then run this command again."
        )
        return

    # Find audio files
    audio_files = sorted(
        f
        for f in speaker_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        _LOGGER.error(
            "No audio files found in %s. "
            "Place WAV files (16kHz, mono) in this directory and try again.",
            speaker_dir,
        )
        sys.exit(1)

    _LOGGER.info(
        "Enrolling '%s' — found %d audio file(s) in %s",
        speaker_name,
        len(audio_files),
        speaker_dir,
    )

    # Load model
    _LOGGER.info("Loading ECAPA-TDNN model...")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        _LOGGER.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    run_opts = {"device": device} if device == "cuda" else {}
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=f"{args.model_dir}/spkrec-ecapa-voxceleb",
        run_opts=run_opts,
    )

    # Extract embeddings from each sample
    embeddings = []
    for audio_file in audio_files:
        _LOGGER.info("Processing: %s", audio_file.name)
        try:
            signal, sample_rate = torchaudio.load(str(audio_file))

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                _LOGGER.info(
                    "  Resampling from %d Hz to 16000 Hz", sample_rate
                )
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                signal = resampler(signal)

            # Convert to mono if needed
            if signal.shape[0] > 1:
                _LOGGER.info("  Converting to mono")
                signal = signal.mean(dim=0, keepdim=True)

            duration = signal.shape[1] / 16000
            _LOGGER.info("  Duration: %.1f seconds", duration)

            if duration < 1.0:
                _LOGGER.warning(
                    "  Skipping — audio is too short (< 1 second)"
                )
                continue

            with torch.no_grad():
                embedding = classifier.encode_batch(signal)

            emb_np = embedding.squeeze().cpu().numpy()
            embeddings.append(emb_np)
            _LOGGER.info("  Embedding extracted (shape=%s)", emb_np.shape)

        except Exception:
            _LOGGER.exception("  Failed to process %s", audio_file.name)
            continue

    if not embeddings:
        _LOGGER.error("No valid embeddings extracted. Check your audio files.")
        sys.exit(1)

    # Average all embeddings to create the voiceprint
    voiceprint = np.mean(embeddings, axis=0)

    # Normalize the voiceprint (unit vector for cosine similarity)
    norm = np.linalg.norm(voiceprint)
    if norm > 0:
        voiceprint = voiceprint / norm

    # Save
    voiceprints_dir.mkdir(parents=True, exist_ok=True)
    output_path = voiceprints_dir / f"{speaker_name}.npy"
    np.save(str(output_path), voiceprint)

    _LOGGER.info(
        "Voiceprint for '%s' saved to %s (from %d sample(s))",
        speaker_name,
        output_path,
        len(embeddings),
    )


if __name__ == "__main__":
    main()
```

### scripts/test_verify.py

```python
"""Test script — verify a WAV file against enrolled voiceprints.

Usage:
    python -m scripts.test_verify /path/to/test.wav [--threshold 0.20]

Useful for tuning the similarity threshold before deploying.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Test speaker verification against a WAV file."""
    parser = argparse.ArgumentParser(
        description="Test a WAV file against enrolled voiceprints"
    )
    parser.add_argument(
        "audio_file",
        help="Path to a WAV file to verify",
    )
    parser.add_argument(
        "--voiceprints-dir",
        default=os.environ.get("VOICEPRINTS_DIR", "/data/voiceprints"),
        help="Directory containing voiceprint .npy files",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/data/models"),
        help="Directory with cached speaker model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("VERIFY_THRESHOLD", "0.20")),
        help="Similarity threshold",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        _LOGGER.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    voiceprints_dir = Path(args.voiceprints_dir)
    if not voiceprints_dir.exists():
        _LOGGER.error("Voiceprints directory not found: %s", voiceprints_dir)
        sys.exit(1)

    # Load voiceprints
    voiceprints = {}
    for npy_file in sorted(voiceprints_dir.glob("*.npy")):
        voiceprints[npy_file.stem] = np.load(str(npy_file))

    if not voiceprints:
        _LOGGER.error("No voiceprints found in %s", voiceprints_dir)
        sys.exit(1)

    # Load model
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    run_opts = {"device": device} if device == "cuda" else {}
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=f"{args.model_dir}/spkrec-ecapa-voxceleb",
        run_opts=run_opts,
    )

    # Load and process audio
    signal, sample_rate = torchaudio.load(str(audio_path))

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        signal = resampler(signal)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    duration = signal.shape[1] / 16000

    # Extract embedding
    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    emb = embedding.squeeze().cpu().numpy()

    # Compare against all voiceprints
    best_score = -1.0
    best_speaker = None

    print()
    print(f"  Audio file:  {audio_path.name}")
    print(f"  Duration:    {duration:.1f}s")
    print(f"  Threshold:   {args.threshold:.2f}")
    print()
    print(f"  {'Speaker':<20} {'Similarity':>10}   Result")
    print(f"  {'─' * 20} {'─' * 10}   {'─' * 10}")

    for name, voiceprint in sorted(voiceprints.items()):
        similarity = 1.0 - cosine(emb, voiceprint)
        is_match = similarity >= args.threshold
        marker = "✓ MATCH" if is_match else "✗"
        print(f"  {name:<20} {similarity:>10.4f}   {marker}")

        if similarity > best_score:
            best_score = similarity
            best_speaker = name

    overall_match = best_score >= args.threshold
    print()
    print(
        f"  Best match:  {best_speaker} ({best_score:.4f}) → "
        f"{'✓ ACCEPTED' if overall_match else '✗ REJECTED'}"
    )
    print()


if __name__ == "__main__":
    main()
```

### tools/record_samples.ps1

```powershell
<#
.SYNOPSIS
    Records voice enrollment samples for Wyoming Voice Match.
.DESCRIPTION
    Lists available microphones, lets you pick one, and records 
    WAV samples into the enrollment folder for a given speaker.
.PARAMETER Speaker
    Name of the speaker to enroll (e.g., "john").
.PARAMETER Samples
    Number of samples to record (default: 7).
.PARAMETER Duration
    Duration of each sample in seconds (default: 5).
.EXAMPLE
    .\record_samples.ps1 -Speaker john
    .\record_samples.ps1 -Speaker jane -Samples 10 -Duration 8
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Speaker,

    [int]$Samples = 30,

    [int]$Duration = 5
)

$ErrorActionPreference = "Stop"

# Check ffmpeg is installed
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found. Install it with: winget install ffmpeg" -ForegroundColor Red
    exit 1
}

# Get audio devices from ffmpeg
Write-Host "`nDetecting audio devices..." -ForegroundColor Cyan
$rawOutput = & cmd /c "ffmpeg -sources dshow 2>&1"

# Parse audio device names and paths
$devices = @()
$devicePaths = @()
$lines = if ($rawOutput -is [array]) { $rawOutput } else { $rawOutput -split "`n" }
foreach ($line in $lines) {
    $lineStr = "$line"
    if ($lineStr -match '^\s+(\S+)\s+\[(.+?)\]\s*\(audio\)') {
        $devicePaths += $Matches[1]
        $devices += $Matches[2]
    }
}

# Fallback: try legacy command if no devices found
if ($devices.Count -eq 0) {
    $rawOutput = & cmd /c "ffmpeg -list_devices true -f dshow -i dummy 2>&1"
    $inAudio = $false
    $lines = if ($rawOutput -is [array]) { $rawOutput } else { $rawOutput -split "`n" }
    foreach ($line in $lines) {
        $lineStr = "$line"
        if ($lineStr -match 'DirectShow audio devices') { $inAudio = $true; continue }
        if ($lineStr -match 'DirectShow video devices') { $inAudio = $false; continue }
        if ($inAudio -and $lineStr -match '"(.+)"') {
            $devices += $Matches[1]
            $devicePaths += $Matches[1]
        }
    }
}

if ($devices.Count -eq 0) {
    Write-Host "No audio devices found. Make sure a microphone is connected." -ForegroundColor Red
    exit 1
}

# Display device list
Write-Host "`nAvailable microphones:" -ForegroundColor Green
for ($i = 0; $i -lt $devices.Count; $i++) {
    Write-Host "  [$($i + 1)] $($devices[$i])"
}

# User selection
do {
    $selection = Read-Host "`nSelect a microphone (1-$($devices.Count))"
} while (-not ($selection -as [int]) -or [int]$selection -lt 1 -or [int]$selection -gt $devices.Count)

$micName = $devices[[int]$selection - 1]
$micPath = $devicePaths[[int]$selection - 1]
Write-Host "`nUsing: $micName" -ForegroundColor Green

# Create enrollment directory relative to script location (tools/../data/enrollment)
$enrollDir = Join-Path $PSScriptRoot "..\data\enrollment\$Speaker"
if (-not (Test-Path $enrollDir)) {
    New-Item -ItemType Directory -Path $enrollDir -Force | Out-Null
}

Write-Host "`nRecording $Samples samples ($Duration seconds each) for speaker '$Speaker'" -ForegroundColor Cyan
Write-Host "Speak naturally - vary your volume and distance from the mic.`n"

$phrases = @(
    "Hey, turn on the living room lights and set them to fifty percent",
    "What is the weather going to be like tomorrow morning",
    "Set a timer for ten minutes and remind me to check the oven",
    "Play some jazz music in the kitchen please",
    "Good morning, what is on my calendar for today",
    "Lock the front door and turn off all the lights downstairs",
    "What is the temperature inside the house right now",
    "Turn the thermostat up to seventy two degrees",
    "Add milk and eggs to my shopping list",
    "Dim the bedroom lights to twenty percent",
    "What time is my first meeting tomorrow",
    "Turn off the TV in the living room",
    "Set an alarm for seven thirty in the morning",
    "Open the garage door",
    "Tell me a joke",
    "How long is my commute to work today",
    "Play my morning playlist on the bedroom speaker",
    "Is the back door locked",
    "Remind me to call the dentist at noon",
    "What is the humidity outside right now",
    "Turn on the fan in the office",
    "Cancel all my alarms for tomorrow",
    "Start the robot vacuum in the living room",
    "How many steps have I taken today",
    "Read me the latest news headlines",
    "Set the lights to warm white in the dining room",
    "Is there any rain expected this weekend",
    "Pause the music for a moment please",
    "Turn on do not disturb mode",
    "Show me the front door camera"
)

# Find existing samples
$existing = Get-ChildItem -Path $enrollDir -Filter "*.wav" -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Found $($existing.Count) existing sample(s) in folder. New samples will be added." -ForegroundColor Cyan
}

for ($i = 0; $i -lt $Samples; $i++) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outFile = Join-Path $enrollDir "${Speaker}_${timestamp}.wav"
    $phrase = $phrases[$i % $phrases.Count]

    Write-Host "[$($i + 1)/$Samples] Say: `"$phrase`"" -ForegroundColor Yellow
    Write-Host "  Recording in 2 seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 2

    Write-Host "  Recording..." -ForegroundColor Red
    & ffmpeg -y -f dshow -i "audio=$micPath" -ar 16000 -ac 1 -t $Duration $outFile -loglevel quiet 2>$null

    if (Test-Path $outFile) {
        Write-Host "  Saved: $outFile" -ForegroundColor Green
    } else {
        Write-Host "  Failed to record sample" -ForegroundColor Red
    }

    if ($i -lt ($Samples - 1)) {
        Start-Sleep -Seconds 1
    }
}

Write-Host "`nDone! Recorded $Samples samples in: $enrollDir" -ForegroundColor Cyan
Write-Host "Now run enrollment to generate the voiceprint:" -ForegroundColor Cyan
Write-Host "  docker compose run --rm wyoming-voice-match python -m scripts.enroll --speaker $Speaker" -ForegroundColor White
```

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

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
    # Remove nvidia pip packages that duplicate libs in the CUDA 12.4+cuDNN runtime base
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cublas && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cudnn && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cufft && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/curand && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cusolver && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/cusparse && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/nccl && \
    rm -rf /usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink && \
    # Remove duplicate CUDA libs from torch/lib (keep cusparseLt)
    cd /usr/local/lib/python3.10/dist-packages/torch/lib && \
    rm -f libnccl* libcublas* libcublasLt* libcusolver* \
          libcufft* libcurand* libnvrtc* libnvJitLink* libnvfuser* && \
    # Remove static libs and test dirs
    find /usr/local/lib/python3.10/dist-packages/torch -name "*.a" -delete && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/include && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/share && \
    # Remove unused pip deps
    pip uninstall -y sympy networkx 2>/dev/null; \
    # Remove SpeechBrain extras
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/recipes && \
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/tests && \
    # Clean caches
    find /usr/local/lib/python3.10/dist-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; \
    echo "Cleanup complete"

# --- Runtime stage ---
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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
```

### Dockerfile.cpu

```dockerfile
FROM python:3.11-slim

LABEL maintainer="Wyoming Voice Match"
LABEL description="Wyoming ASR proxy with ECAPA-TDNN speaker verification (CPU-only)"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch)
COPY requirements.cpu.txt .
RUN pip install --no-cache-dir -r requirements.cpu.txt

# Copy application code
COPY wyoming_voice_match/ wyoming_voice_match/
COPY scripts/ scripts/

# Create data directory structure
RUN mkdir -p /data/enrollment /data/voiceprints /data/models

EXPOSE 10350

ENTRYPOINT ["python", "-m", "wyoming_voice_match"]
```

### requirements.txt

```
wyoming==1.8.0
speechbrain>=1.0.0
scipy>=1.11.0
numpy>=1.24.0
requests>=2.28.0
huggingface_hub<0.27.0
```

### requirements.cpu.txt

```
wyoming==1.8.0
speechbrain>=1.0.0
torch>=2.1.0
torchaudio>=2.1.0
scipy>=1.11.0
numpy>=1.24.0
requests>=2.28.0
huggingface_hub<0.27.0
```

### docker-compose.yml

```yaml
services:
  wyoming-voice-match:
    image: jxlarrea/wyoming-voice-match:latest
    container_name: wyoming-voice-match
    restart: unless-stopped
    ports:
      - "10350:10350"
    volumes:
      - ./data:/data
    environment:
      - UPSTREAM_URI=tcp://wyoming-faster-whisper:10300
      - VERIFY_THRESHOLD=0.20
      - LISTEN_URI=tcp://0.0.0.0:10350
      - DEVICE=cuda
      - HF_HOME=/data/hf_cache
      - LOG_LEVEL=DEBUG
      # - MAX_VERIFY_SECONDS=5.0   # First-pass verification window
      # - VERIFY_WINDOW_SECONDS=3.0       # Sliding window size for fallback pass
      # - VERIFY_STEP_SECONDS=1.5         # Sliding window step size
      # - ASR_MAX_SECONDS=3.0     # Max audio duration forwarded to ASR
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### docker-compose.cpu.yml

```yaml
services:
  wyoming-voice-match:
    image: jxlarrea/wyoming-voice-match:cpu
    container_name: wyoming-voice-match
    restart: unless-stopped
    ports:
      - "10350:10350"
    volumes:
      - ./data:/data
    environment:
      - UPSTREAM_URI=tcp://wyoming-faster-whisper:10300
      - VERIFY_THRESHOLD=0.20
      - LISTEN_URI=tcp://0.0.0.0:10350
      - DEVICE=cpu
      - HF_HOME=/data/hf_cache
      - LOG_LEVEL=DEBUG
      # - MAX_VERIFY_SECONDS=5.0   # First-pass verification window
      # - VERIFY_WINDOW_SECONDS=3.0       # Sliding window size for fallback pass
      # - VERIFY_STEP_SECONDS=1.5         # Sliding window step size
      # - ASR_MAX_SECONDS=3.0     # Max audio duration forwarded to ASR
```

### LICENSE

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```