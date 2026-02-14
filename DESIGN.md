# Design & Architecture

Complete technical specification for Wyoming Voice Match. This document contains enough detail to rebuild the project from scratch.

## Table of Contents

- [Project Structure](#project-structure)
- [Core Concept](#core-concept)
- [Dependencies](#dependencies)
- [Pipeline Architecture](#pipeline-architecture)
  - [Two Execution Paths](#two-execution-paths)
  - [Speaker Verification (Three-Pass Strategy)](#speaker-verification-three-pass-strategy)
  - [Speaker Extraction (Two-Stage)](#speaker-extraction-two-stage)
  - [AudioStop Synchronization](#audiostop-synchronization)
- [Entry Point (__main__.py)](#entry-point-__main__py)
- [Handler (handler.py)](#handler-handlerpy)
- [Enhancer (enhance.py)](#enhancer-enhancepy)
- [Verifier (verify.py)](#verifier-verifypy)
- [Enrollment (scripts/enroll.py)](#enrollment-scriptsenrollpy)
- [Satellite Enrollment Recording (scripts/enroll_record.py)](#satellite-enrollment-recording-scriptsenroll_recordpy)
- [Test Script (scripts/test_verify.py)](#test-script-scriptstest_verifypy)
- [Demo Script (scripts/demo.py)](#demo-script-scriptsdemopy)
- [Docker Images](#docker-images)
- [Concurrency Model](#concurrency-model)
- [Observed Performance Characteristics](#observed-performance-characteristics)
- [Design Decisions & Rationale](#design-decisions--rationale)
- [Appendix: Complete Source Files](#appendix-complete-source-files)

## Project Structure

```
wyoming-voice-match/
├── wyoming_voice_match/          # Main Python package
│   ├── __init__.py               # Version string (__version__ = "1.0.0")
│   ├── __main__.py               # Entry point, arg parsing, server setup
│   ├── handler.py                # Wyoming event handler (ASR proxy logic)
│   ├── enhance.py                # Optional speech enhancement (SepFormer denoising)
│   └── verify.py                 # ECAPA-TDNN speaker verification + extraction
├── scripts/
│   ├── __init__.py               # Empty, makes scripts a package
│   ├── demo.py                   # Pipeline demo CLI (verify + extract a WAV file)
│   ├── enroll.py                 # Voice enrollment CLI
│   ├── enroll_record.py          # Record enrollment samples from a satellite
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
├── README.md                     # User-facing documentation
└── DESIGN.md                     # This file
```

## Core Concept

Wyoming Voice Match is a Wyoming protocol ASR proxy that sits between Home Assistant and an ASR (speech-to-text) service. It solves the "TV problem": when a satellite microphone picks up both the user's voice and TV/radio audio, the ASR gets garbage transcripts mixing voice commands with TV dialogue.

The solution has two parts:
1. **Speaker verification** - confirm the audio contains an enrolled speaker's voice (rejects TV-only triggers)
2. **Speaker extraction** - isolate only the enrolled speaker's voice segments from the full audio buffer, discarding TV/radio/other people before sending to ASR

## Dependencies

### Python Packages

**requirements.txt** (GPU - torch/torchaudio installed separately from `https://download.pytorch.org/whl/cu121`):
```
wyoming==1.8.0
speechbrain>=1.0.0
scipy>=1.11.0
numpy>=1.24.0
requests>=2.28.0
huggingface_hub<0.27.0
```

**requirements.cpu.txt** (CPU - torch NOT included here, installed separately in Dockerfile.cpu with pinned versions):
```
wyoming==1.8.0
speechbrain>=1.0.0
scipy>=1.11.0
numpy>=1.24.0
requests>=2.28.0
huggingface_hub<0.27.0
```

Key dependency notes:
- `huggingface_hub<0.27.0` - pinned to avoid a breaking change in 0.27+ that affects SpeechBrain's model loading
- `wyoming==1.8.0` - the Wyoming protocol library (AsyncServer, AsyncEventHandler, AsyncClient, event types)
- `speechbrain` - provides `EncoderClassifier` for ECAPA-TDNN inference
- `scipy` - only used for `scipy.spatial.distance.cosine`
- `torchaudio` - only used in enrollment/test scripts for loading WAV files (the main service receives raw PCM bytes)

### System Packages (in Docker)

- `libsndfile1` - required by SpeechBrain/torchaudio for audio I/O
- `ffmpeg` - used by enrollment script for format conversion
- `libgomp1` - OpenMP runtime (GPU image only, required by torch)
- `soundfile` - Python package, CPU image only, provides audio backend for torchaudio

## Pipeline Architecture

### Two Execution Paths

The handler has two execution paths depending on audio length:

**Early Pipeline** (`_run_early_pipeline`) - triggered when buffer reaches `MAX_VERIFY_SECONDS` (default 5.0s):
1. Verify speaker identity using first 5s snapshot
2. If verified, set `_responded = True` and wait for AudioStop
3. When AudioStop arrives, run speaker extraction on full buffer
4. Forward extracted audio to ASR
5. Return transcript to Home Assistant

**Sync Pipeline** (`_process_audio_sync`) - triggered at AudioStop if early pipeline hasn't responded:
1. Used for short audio (< 5s) or when early verify rejected but full-audio re-verify passes
2. Verify speaker on full buffer
3. If verified and audio > 3s, run speaker extraction to remove background noise
4. Forward extracted audio to ASR
5. Return transcript or empty string

### Speaker Verification (Three-Pass Strategy)

The `verify()` method uses multiple passes to handle noisy environments:

**Pass 1 - Speech segments (multi-candidate):** Energy analysis finds up to 3 energy peaks in the audio, ranked by intensity. In noisy environments, the loudest peak may be TV or background audio rather than the user's voice. Each candidate segment is verified against the voiceprint in order — if the first (loudest) rejects, the next peak is tried. This allows the system to find the user's voice even when it's not the loudest sound in the room. If any candidate matches → accept.

**Pass 2 - First-N seconds:** If pass 1 fails, verify the first `MAX_VERIFY_SECONDS` of audio as a single chunk. This works when the voice is distributed across the audio rather than concentrated. If it matches → accept.

**Pass 3 - Sliding window:** If passes 1 and 2 fail, scan the full audio with overlapping windows (default 3.0s window, 1.5s step). This catches speech that may be at an arbitrary position. Best score wins.

The speech segment from pass 1 is preserved in the `VerificationResult` for use by the extraction stage.

### Speaker Extraction (Two-Stage)

After verification passes and AudioStop arrives, `extract_speaker_audio()` isolates the enrolled speaker's voice:

**Stage 1 - Energy-based speech detection:**
- Split buffer into 50ms frames, compute RMS energy per frame
- Determine noise floor using 10th percentile of frame energies × 5 (minimum 500)
- Find contiguous regions of high-energy frames (with 300ms gap bridging)
- Output: list of (start_frame, end_frame) speech regions

**Stage 2 - Voiceprint verification per region:**
- For each speech region, ensure minimum 1s length (expand symmetrically if shorter)
- Extract ECAPA-TDNN embedding from the region
- Compare against enrolled speaker's voiceprint via cosine similarity
- Keep regions with similarity ≥ `threshold * 0.5` (half the verification threshold, more lenient)
- If a region is 3s+ and **fails** the check, run rescue scanning (see Stage 3) - the speaker's voice may be buried inside a larger blob of background audio

**Stage 3 - Sub-region scanning (for regions 3s+):**

Applies in two cases:

*Edge trimming* - when a long region passes Stage 2, a sliding window (1.5s window, 0.5s step) scans the entire region. The longest contiguous run of windows matching the speaker determines the trim boundaries. The trim threshold is the midpoint between the extraction threshold and the region's overall similarity score, which adapts to the actual signal quality. This removes TV/background audio that got bundled at the edges because energy levels never dropped long enough to create a gap.

*Rescue scanning* - when a long region fails Stage 2 (the speaker's voice is diluted by surrounding background audio), the same sliding window scans for sub-segments that match. Uses the full verification threshold (`self.threshold`) since a genuine voice window should score near verification levels. This rescues short commands (e.g., "what time is it") spoken over continuous background noise.

The final output is the concatenation of all kept (and potentially trimmed/rescued) regions.

**Why this works:** The user's voice near the mic produces embeddings matching their voiceprint (similarity 0.20-0.55). TV speakers produce completely different embeddings (similarity -0.04 to 0.07). The voiceprint comparison cleanly separates these regardless of volume overlap. The sub-region scanning handles cases where energy analysis merges the speaker's voice with background audio into a single region.

**Key parameters:**
- Frame size for energy analysis: 50ms
- Gap bridging: 300ms (joins speech regions separated by brief pauses)
- Minimum region length for embedding: 1.0s (shorter segments produce unreliable embeddings)
- Extraction similarity threshold: `EXTRACTION_THRESHOLD` (default 0.25)
- Sub-region scan minimum region length: 3.0s
- Sub-region scan window: 1.5s, step: 0.5s
- Trim threshold: midpoint of extraction threshold and region similarity
- Rescue threshold: `VERIFY_THRESHOLD` (full verification threshold, default 0.30)
- Noise floor: 10th percentile of frame RMS × 5, minimum 500

### AudioStop Synchronization

Critical implementation detail: the early pipeline must wait for the full audio stream before running extraction. This is achieved with `asyncio.Event`:

- `_audio_stopped = asyncio.Event()` - created fresh per session in AudioStart handler
- Set by the AudioStop handler (always, regardless of `_responded` state)
- Awaited by `_run_early_pipeline` with 30s timeout
- Audio chunks continue buffering after `_responded = True` (important: the AudioChunk handler must NOT skip buffering when responded)

## Entry Point (__main__.py)

### Argument Parsing

All arguments have environment variable fallbacks for Docker configuration:

| CLI Flag | Env Var | Default | Description |
|---|---|---|---|
| `--upstream-uri` | `UPSTREAM_URI` | `tcp://localhost:10300` | Upstream ASR service URI |
| `--uri` | `LISTEN_URI` | `tcp://0.0.0.0:10350` | Wyoming server listen URI |
| `--threshold` | `VERIFY_THRESHOLD` | `0.30` | Cosine similarity threshold |
| `--extraction-threshold` | `EXTRACTION_THRESHOLD` | `0.25` | Extraction similarity threshold |
| `--require-speaker-match` | `REQUIRE_SPEAKER_MATCH` | `true` | When `false`, unmatched audio is forwarded instead of rejected — enrolled speakers still get extraction |
| `--tag-speaker` | `TAG_SPEAKER` | `false` | Prepend `[speaker_name]` to transcripts |
| `--isolate-voice` | `ISOLATE_VOICE` | `0` | Voice isolation: 0=disabled, 0.5=moderate, 1.0=full |
| `--debug` | `LOG_LEVEL=DEBUG` | `INFO` | Enable debug logging |
| `--device` | `DEVICE` | `cuda` | `cuda` or `cpu` (auto-detects, falls back to cpu) |
| `--voiceprints-dir` | `VOICEPRINTS_DIR` | `/data/voiceprints` | Directory with .npy voiceprints |
| `--model-dir` | `MODEL_DIR` | `/data/models` | Model cache directory |
| `--max-verify-seconds` | `MAX_VERIFY_SECONDS` | `5.0` | Early verification trigger |
| `--window-seconds` | `VERIFY_WINDOW_SECONDS` | `3.0` | Sliding window size |
| `--step-seconds` | `VERIFY_STEP_SECONDS` | `1.5` | Sliding window step |

### Startup Sequence

1. Parse args
2. Configure logging (DEBUG or INFO)
3. Validate voiceprints directory exists (exit 1 if not)
4. Create `SpeakerVerifier` - loads ECAPA-TDNN model and all .npy voiceprints
5. Validate at least one voiceprint loaded (exit 1 if `require_speaker_match` is true, warn if false)
6. If `isolate_voice` > 0, create `SpeechEnhancer` - loads SepFormer enhancement model
7. Query upstream ASR for supported languages via `query_upstream_languages()`
8. Build `wyoming.info.Info` with ASR program/model metadata and upstream languages
9. Create `AsyncServer` and run with `SpeakerVerifyHandler` factory

The handler factory uses `functools.partial` to pass `wyoming_info`, `verifier`, `upstream_uri`, `tag_speaker`, `require_speaker_match`, and `enhancer` to each new handler instance.

### Wyoming Service Info

The service registers as an ASR program named `"voice-match"` with model `"voice-match-proxy"`. At startup, it queries the upstream ASR service for its supported languages and advertises those same languages to Home Assistant. If the upstream is unreachable, it falls back to an empty language list. This makes it appear as a standard STT service in Home Assistant and ensures it can be assigned to any pipeline the upstream ASR supports.

### query_upstream_languages(uri, timeout) → List[str]

Connects to the upstream ASR, sends a `Describe` event, and reads the `Info` response to collect all supported languages from its models. Returns a deduplicated list preserving order. Returns an empty list on any error (connection refused, timeout, etc.).

## Handler (handler.py)

### Class: SpeakerVerifyHandler

Extends `wyoming.server.AsyncEventHandler`. One instance per TCP connection.

**Module-level state:**
- `_MODEL_LOCK: asyncio.Lock` - prevents concurrent ECAPA-TDNN inference across all handlers

**Instance state:**
- `wyoming_info: Info` - service metadata for Describe responses
- `verifier: SpeakerVerifier` - shared verifier instance
- `upstream_uri: str` - ASR service URI
- `tag_speaker: bool` - whether to prepend `[speaker_name]` to transcripts
- `require_speaker_match: bool` - when false, bypass verification and forward all audio directly
- `enhancer: Optional[SpeechEnhancer]` - speech enhancement model (None if disabled)
- `_audio_buffer: bytes` - accumulated PCM audio (keeps growing even after verification)
- `_audio_rate: int` - sample rate (default 16000)
- `_audio_width: int` - bytes per sample (default 2 = 16-bit)
- `_audio_channels: int` - channel count (default 1 = mono)
- `_language: Optional[str]` - language from Transcribe event
- `_verify_task: Optional[asyncio.Task]` - background verification task
- `_verify_started: bool` - whether early verification was triggered
- `_responded: bool` - whether transcript was already sent
- `_stream_start_time: Optional[float]` - monotonic timestamp for latency tracking
- `_session_id: str` - 8-char hex UUID for log correlation
- `_audio_stopped: asyncio.Event` - signals that AudioStop was received

### Event Handling Flow

```
handle_event(event) dispatches by event type:

Describe → write Info event back (service discovery)
Transcribe → store language preference
AudioStart → reset all per-stream state (including _audio_stopped)
AudioChunk → ALWAYS append to buffer, check early verify trigger if not yet started
AudioStop → set _audio_stopped event, then either log (if responded) or call _process_audio_sync
```

**AudioChunk handler (critical detail):**
1. Always append chunk audio to `_audio_buffer` regardless of `_responded` state
2. Only check early verification trigger if `not _verify_started and not _responded`
3. If `buffered_seconds >= max_verify_seconds`: set `_verify_started = True`, snapshot buffer, create `_run_early_pipeline` task

**AudioStop handler:**
1. Always set `_audio_stopped.set()` (even if already responded)
2. If `_responded` is True, log and return
3. Otherwise call `_process_audio_sync()` (verifies and forwards; if rejected and `require_speaker_match` is false, forwards unmodified audio instead of returning empty transcript)

### _run_early_pipeline(verify_audio: bytes)

Called as a background task when 5s of audio is buffered.

1. Acquire `_MODEL_LOCK`
2. Run `verifier.verify(verify_audio, sample_rate)` in `run_in_executor`
3. Release lock
4. If rejected:
   - Cache result in `_verify_result_cache`
   - Return (wait for AudioStop to try full audio)
5. If matched:
   - Log speaker name, similarity, threshold
   - Set `_responded = True` immediately (prevents AudioStop from triggering sync path)
   - Await `_audio_stopped.wait()` with 30s timeout (waits for full stream)
   - Snapshot full buffer
   - Acquire `_MODEL_LOCK`
   - Run `verifier.extract_speaker_audio(full_buffer, speaker_name, sample_rate)` in `run_in_executor`
   - Release lock
   - Forward extracted audio to upstream ASR with `_forward_to_upstream()`
   - Apply speaker tagging if enabled: `_tag_transcript(transcript, speaker_name)` prepends `[speaker_name]`
   - Write `Transcript` event back to HA
   - Log total pipeline time with breakdown (verify, extract, asr, processing)

### _process_audio_sync()

Fallback path when early verification hasn't responded (short audio or early verify rejected).

1. If empty buffer → return empty transcript
2. Check `_verify_result_cache` (early verify rejected, re-try with full audio)
3. If no cache → first-time verification on full buffer
4. If matched and audio > 3s → run speaker extraction, forward extracted audio to ASR, apply speaker tagging if enabled
5. If matched and audio ≤ 3s → forward full buffer to ASR (too short for meaningful extraction)
6. If rejected and `require_speaker_match` is true → return empty transcript
7. If rejected and `require_speaker_match` is false → forward full buffer unmodified to ASR

### _forward_to_upstream(audio_bytes: bytes) → str

Sends audio to the upstream Wyoming ASR service:

1. Open `AsyncClient` connection to `upstream_uri`
2. Send `Transcribe` event (with language if set)
3. Send `AudioStart` event (rate, width, channels)
4. Stream audio in 100ms chunks
5. Send `AudioStop`
6. Read events until `Transcript` received
7. Return transcript text (or empty string on error)

### _tag_transcript(transcript: str, speaker_name: Optional[str]) → str

Prepends `[speaker_name]` to the transcript when `tag_speaker` is enabled. Returns the transcript unchanged if tagging is disabled, speaker name is None, or transcript is empty. Useful for LLM-based conversation agents that can use the speaker identity for personalization.

### _maybe_enhance(audio_bytes: bytes, sid: str) → (bytes, float)

If `self.enhancer` is set (ISOLATE_VOICE > 0), runs speech enhancement on the audio under `_MODEL_LOCK` and returns `(enhanced_bytes, enhance_ms)`. If enhancer is None, returns the input unchanged with `enhance_ms=0.0`. Called after extraction and before `_forward_to_upstream()` in both the early pipeline and sync paths.

## Enhancer (enhance.py)

### Class: SpeechEnhancer

Optional voice isolation module that removes residual background noise from extracted speaker audio before forwarding to ASR. Uses a SepFormer model via SpeechBrain. Language-agnostic — operates on acoustic features, not linguistic content.

Controlled by `ISOLATE_VOICE` (0.0–1.0): 0 disables it entirely (no model loaded), values between 0 and 1 blend the original and enhanced signals, and 1.0 applies full SepFormer enhancement.

**Constructor parameters:**
- `model_dir: str` - directory to cache downloaded model weights
- `device: str` - "cuda" or "cpu" (auto-falls-back to cpu)
- `model_source: str` - HuggingFace model identifier (default: `speechbrain/sepformer-wham16k-enhancement`)
- `isolate_voice: float` - blend level between original and enhanced (0.0–1.0)

**On construction:**
1. Import `SepformerSeparation` from SpeechBrain (deferred to avoid loading when disabled)
2. Auto-detect CUDA availability, fall back to CPU if needed
3. Load pretrained SepFormer enhancement model (~113 MB)

### enhance(audio_bytes, sample_rate, sample_width) → bytes

1. Convert raw 16-bit PCM bytes to float32 tensor in [-1.0, 1.0] range
2. Run SepFormer inference (`separate_batch`) — outputs enhanced waveform
3. Blend enhanced with original based on `isolate_voice` level
4. Clamp and convert back to 16-bit PCM bytes

**Pipeline position:** After speaker extraction, before ASR forwarding. Only runs when a speaker was matched (not on bypass/unmatched audio).

## Verifier (verify.py)

### Class: SpeakerVerifier

**Constructor parameters:**
- `voiceprints_dir: str` - path to directory of .npy files
- `model_dir: str` - HuggingFace model cache
- `device: str` - "cuda" or "cpu" (auto-falls-back to cpu if CUDA unavailable)
- `threshold: float` - cosine similarity threshold (default 0.20)
- `max_verify_seconds: float` - controls early verification trigger and first-N pass length
- `window_seconds: float` - sliding window size for pass 3
- `step_seconds: float` - sliding window step for pass 3

**On construction:**
1. Auto-detect CUDA: if device="cuda" but `torch.cuda.is_available()` is False → fall back to "cpu" with warning
2. Load ECAPA-TDNN via `EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", ...)`
3. Load all .npy voiceprints from the directory

### VerificationResult dataclass

```python
@dataclass
class VerificationResult:
    is_match: bool
    similarity: float
    threshold: float
    matched_speaker: Optional[str] = None
    all_scores: Dict[str, float] = field(default_factory=dict)
    speech_audio: Optional[bytes] = field(default=None, repr=False)
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None
```

### Key Methods

**`verify(audio_bytes, sample_rate) → VerificationResult`**
Three-pass verification as described in Pipeline Architecture. Returns best result across all passes.

**`extract_speaker_audio(audio_bytes, speaker_name, sample_rate, similarity_threshold?) → bytes`**
Three-stage speaker extraction as described in Pipeline Architecture. Returns concatenated PCM of kept (and potentially trimmed/rescued) regions.

**`_trim_region(audio_bytes, start_frame, end_frame, ..., similarity_threshold, window_seconds, step_seconds, frame_ms) → Optional[bytes]`**
Sliding window scan used for both edge trimming and rescue scanning. Scans the entire region, finds the longest contiguous run of windows matching the threshold, and returns only that portion. Returns None if no matches found.

**`_extract_speech_segment(audio_bytes, sample_rate) → Optional[Tuple[bytes, float, float]]`**
Convenience wrapper — returns the top candidate from `_extract_speech_candidates()`.

**`_extract_speech_candidates(audio_bytes, sample_rate, max_candidates=3) → List[Tuple[bytes, float, float]]`**
Multi-candidate energy-based speech detection for verification pass 1:
- 50ms frames, compute RMS energy per frame
- Find up to 3 energy peaks by iteratively selecting the loudest frame, expanding outward while energy stays above 15% of peak, then masking that region
- Each candidate is expanded to minimum 1s
- Returns list of (segment_bytes, start_sec, end_sec) ranked by energy
- In noisy environments, the loudest peak may be TV — subsequent candidates allow pass 1 to find the speaker's voice even when it's quieter than the background

**`_verify_chunk(audio_bytes, sample_rate) → VerificationResult`**
Core verification: extract embedding, compare against all voiceprints, return best match.

**`_extract_embedding(audio_bytes, sample_rate) → np.ndarray`**
Convert PCM to normalized float tensor, run through `classifier.encode_batch()`, return 192-dim numpy array.

**`extract_embedding(audio_bytes, sample_rate) → np.ndarray`**
Public wrapper of `_extract_embedding` for enrollment scripts.

## Enrollment (scripts/enroll.py)

CLI tool that generates voiceprints from WAV samples.

**Usage:**
```bash
python -m scripts.enroll --speaker john    # Enroll from WAV files
python -m scripts.enroll --list            # List enrolled speakers
python -m scripts.enroll --delete john     # Delete voiceprint
```

**Enrollment process:**
1. Find audio files in `enrollment/<speaker>/` (supports .wav, .flac, .ogg, .mp3)
2. Load ECAPA-TDNN model
3. For each file: load audio, resample to 16kHz mono, extract embedding
4. Average all embeddings: `np.mean(embeddings, axis=0)`
5. L2-normalize the average: `voiceprint / np.linalg.norm(voiceprint)`
6. Save as `voiceprints/<speaker>.npy`

**Key detail:** The voiceprint is L2-normalized (unit vector). This means cosine similarity reduces to a simple dot product during verification.

## Satellite Enrollment Recording (scripts/enroll_record.py)

CLI tool that records enrollment samples directly from a Wyoming satellite. This exists because satellite microphones (e.g., Home Assistant Voice PE) have different audio characteristics than PC/phone microphones. A voiceprint enrolled from PC recordings may not match well when the satellite streams audio. Recording through the satellite ensures the enrollment samples match the exact hardware the speaker will use.

**Usage:**
```bash
# Stop the main service first (both use the same port):
docker compose stop wyoming-voice-match

python -m scripts.enroll_record --speaker john --samples 10

# Restart afterward:
docker compose start wyoming-voice-match
```

**How it works:**
1. Starts a Wyoming ASR server on the configured listen port
2. Registers as an ASR service so the satellite can connect
3. For each sample: satellite wake word triggers audio streaming, handler buffers all audio until AudioStop, saves as WAV in `enrollment/<speaker>/satellite_NN.wav`
4. Responds with an empty `Transcript` event so the satellite plays its done sound without triggering a conversation response
5. After all samples are collected, automatically runs `scripts.enroll` as a subprocess to generate the voiceprint
6. Exits

**Architecture:**
- `EnrollRecordState` - shared state across handler instances: speaker name, target sample count, samples recorded counter, and an `asyncio.Event` to signal completion
- `EnrollRecordHandler` - extends `AsyncEventHandler`, buffers audio chunks, saves WAV on AudioStop, sends progress transcript
- Server runs as a background task; main coroutine awaits the done event, then cancels the server and runs enrollment

**Arguments:**

| CLI Flag | Env Var | Default | Description |
|---|---|---|---|
| `--speaker` | - | (required) | Speaker name |
| `--samples` | - | `5` | Number of samples to record |
| `--uri` | `LISTEN_URI` | `tcp://0.0.0.0:10350` | Wyoming server listen URI |
| `--enrollment-dir` | `ENROLLMENT_DIR` | `/data/enrollment` | Root enrollment directory |
| `--voiceprints-dir` | `VOICEPRINTS_DIR` | `/data/voiceprints` | Voiceprint output directory |
| `--model-dir` | `MODEL_DIR` | `/data/models` | Model cache directory |
| `--device` | `DEVICE` | `cuda` | Inference device |

## Test Script (scripts/test_verify.py)

CLI tool for tuning the similarity threshold:
```bash
python -m scripts.test_verify /path/to/test.wav --threshold 0.20
```

Loads a WAV file, extracts an embedding, and compares against all enrolled voiceprints. Displays a table of similarities with MATCH/REJECT indicators.

## Demo Script (scripts/demo.py)

CLI tool that runs the full verification and extraction pipeline on a WAV file, writing the extracted audio as a new WAV:
```bash
docker compose run --rm --entrypoint python wyoming-voice-match \
  -m scripts.demo --speaker john --input /data/test.wav --output /data/cleaned.wav
```

Thresholds are read from `VERIFY_THRESHOLD` and `EXTRACTION_THRESHOLD` environment variables (set in `docker-compose.yml`), matching the main service configuration.

**Arguments:**

| CLI Flag | Description |
|---|---|
| `--input`, `-i` | Input WAV file (any sample rate/channels — auto-converted) |
| `--output`, `-o` | Output WAV file (extracted speaker audio only) |
| `--speaker`, `-s` | Name of the enrolled speaker to extract |
| `--voiceprints-dir` | Directory with .npy voiceprints (default: `/data/voiceprints`) |
| `--model-dir` | Model cache directory (default: `/data/models`) |
| `--device` | `cuda` or `cpu` (default: `cuda`) |

**Process:**
1. Load input WAV (any sample rate or channel count - automatically resampled to 16kHz mono)
2. Run full three-pass speaker verification, displaying similarity scores for all enrolled speakers
3. Run three-stage speaker extraction (energy detection + voiceprint filtering + sub-region scanning)
4. Write extracted audio (only the enrolled speaker's voice) as a WAV file
5. Print summary: input/output duration, amount removed, per-region scores

Debug logging is always enabled so the output shows the complete region-by-region breakdown including individual similarity scores, energy thresholds, and keep/drop decisions. The output WAV represents exactly what would be forwarded to the upstream ASR service during normal operation.

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

Single-stage: `python:3.11-slim`
- Installs libsndfile1, ffmpeg
- Pins torch and torchaudio versions: `torch==2.4.1+cpu torchaudio==2.4.1+cpu` from `https://download.pytorch.org/whl/cpu`
- Installs `soundfile` package (provides audio backend for torchaudio)
- Installs remaining requirements from requirements.cpu.txt
- Same application code and directory structure
- Same entrypoint

**Critical CPU note:** The torch/torchaudio versions must be pinned because the CPU PyTorch index can have stale torchaudio versions that lack `list_audio_backends`, which SpeechBrain requires at import time. The `soundfile` package is also required as the audio backend.

### Data Volume

The `/data` directory is mounted as a Docker volume and contains:
- `/data/enrollment/<speaker>/` - WAV files for each enrolled speaker
- `/data/voiceprints/<speaker>.npy` - generated voiceprint embeddings (192-dim numpy arrays)
- `/data/models/spkrec-ecapa-voxceleb/` - cached ECAPA-TDNN model from HuggingFace
- `/data/hf_cache/` - HuggingFace Hub cache (set via `HF_HOME` env var)

## Concurrency Model

- `AsyncEventHandler` processes events sequentially per connection
- `_MODEL_LOCK` (module-level `asyncio.Lock`) serializes all ECAPA-TDNN inference
- All model inference runs in `run_in_executor` to avoid blocking the event loop
- Audio chunks continue to be buffered by the event loop while model inference runs in background
- `_audio_stopped` event allows the early pipeline to wait for AudioStop without polling

## Observed Performance Characteristics

Based on real-world testing with TV background noise:

- **Speaker verification:** 5-25ms on GPU (cached model), 200-500ms on CPU
- **Speaker extraction:** 15-35ms on GPU for 10-15s buffer (4-6 speech regions)
- **Your voice similarity scores:** 0.20-0.55 (varies with conditions)
- **TV speaker similarity scores:** -0.04 to 0.07 (consistently near zero)
- **Energy threshold (10th percentile × 5):** ~1000-2000 in quiet room, ~2000-4500 with TV
- **Typical pipeline time with TV:** 8-14s (dominated by waiting for AudioStop from satellite)
- **Typical pipeline time in quiet room:** 3-10s (AudioStop arrives quickly from satellite VAD)

## Design Decisions & Rationale

### Why voiceprint extraction instead of energy-based trimming?

Early iterations tried various energy-based approaches to trim TV audio from the buffer:
- **Fixed ASR_MAX_SECONDS cap:** Fails for variable-length commands
- **Speech-end detection (RMS polling):** Fooled by TV pauses triggering false positives
- **Energy threshold scanning:** TV and voice energy overlap in the 3000-12000 RMS range

Energy analysis cannot distinguish voice from TV because they occupy the same energy range. Speaker embeddings can, because they capture voice identity regardless of volume.

### Why wait for AudioStop instead of responding immediately?

Earlier versions forwarded audio immediately after verification (at ~5s), which was fast but limited command length to 5s. The current approach waits for the satellite to finish streaming, then extracts the speaker's voice. This supports commands of any length.

### Why is extraction similarity threshold half the verification threshold?

Verification uses the full threshold (0.20) on clean, loudest-energy speech segments. Extraction needs to be more lenient (0.10) because:
- Speech regions may contain partial words at boundaries
- Quieter syllables produce weaker embeddings
- Missing a region of the user's voice is worse than including a region (Whisper handles imperfect audio well)

### Why L2-normalize voiceprints during enrollment?

Normalizing to unit vectors means `1 - cosine(a, b)` simplifies to a dot product. More importantly, averaging multiple enrollment samples and normalizing produces a robust centroid that's equidistant from all sample embeddings.

---

## Appendix: Complete Source Files

All source files are included below so this document is fully self-contained.

### wyoming_voice_match/__init__.py

```python
"""Wyoming Voice Match - ASR proxy with speaker verification."""

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
from typing import List

import numpy as np

from wyoming.client import AsyncClient
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import SpeakerVerifyHandler
from .verify import SpeakerVerifier

_LOGGER = logging.getLogger(__name__)


async def query_upstream_languages(uri: str, timeout: float = 10.0) -> List[str]:
    """Query the upstream ASR service for its supported languages."""
    try:
        async with AsyncClient.from_uri(uri) as client:
            await client.write_event(Describe().event())
            while True:
                event = await asyncio.wait_for(client.read_event(), timeout=timeout)
                if event is None:
                    break
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    languages = []
                    for asr in info.asr:
                        for model in asr.models:
                            languages.extend(model.languages)
                    seen = set()
                    unique = []
                    for lang in languages:
                        if lang not in seen:
                            seen.add(lang)
                            unique.append(lang)
                    if unique:
                        _LOGGER.info(
                            "Upstream ASR supports %d language(s): %s",
                            len(unique), ", ".join(unique),
                        )
                        return unique
    except Exception as exc:
        _LOGGER.warning("Could not query upstream ASR languages: %s", exc)
    return []


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
        default=float(os.environ.get("VERIFY_THRESHOLD", "0.30")),
        help="Cosine similarity threshold for verification (default: 0.30)",
    )
    parser.add_argument(
        "--extraction-threshold",
        type=float,
        default=float(os.environ.get("EXTRACTION_THRESHOLD", "0.25")),
        help="Cosine similarity threshold for speaker extraction (default: 0.25)",
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
        "--tag-speaker",
        action="store_true",
        default=os.environ.get("TAG_SPEAKER", "false").lower() in ("true", "1", "yes"),
        help="Prepend [speaker_name] to transcripts (default: false)",
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

    voiceprints_dir = Path(args.voiceprints_dir)
    if not voiceprints_dir.exists():
        _LOGGER.error(
            "Voiceprints directory not found at %s. "
            "Run the enrollment script first: "
            "python -m scripts.enroll --speaker <n>",
            voiceprints_dir,
        )
        sys.exit(1)

    _LOGGER.info("Loading ECAPA-TDNN speaker verification model...")
    verifier = SpeakerVerifier(
        voiceprints_dir=str(voiceprints_dir),
        model_dir=args.model_dir,
        device=args.device,
        threshold=args.threshold,
        extraction_threshold=args.extraction_threshold,
        max_verify_seconds=args.max_verify_seconds,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    if not verifier.voiceprints:
        _LOGGER.error(
            "No voiceprints found in %s. "
            "Run the enrollment script first: "
            "python -m scripts.enroll --speaker <n>",
            voiceprints_dir,
        )
        sys.exit(1)

    _LOGGER.info(
        "Speaker verifier ready - %d speaker(s) enrolled "
        "(threshold=%.2f, device=%s, verify_window=%.1fs, "
        "sliding_window=%.1fs/%.1fs)",
        len(verifier.voiceprints),
        args.threshold,
        args.device,
        args.max_verify_seconds,
        args.window_seconds,
        args.step_seconds,
    )

    upstream_languages = await query_upstream_languages(args.upstream_uri)
    if not upstream_languages:
        _LOGGER.warning(
            "Could not detect upstream ASR languages, defaulting to all. "
            "Ensure the upstream ASR is running at %s", args.upstream_uri,
        )

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
                        languages=upstream_languages,
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
            args.tag_speaker,
        )
    )


def run() -> None:
    """Sync wrapper for main."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
```

### wyoming_voice_match/handler.py

See handler.py source in the main project. The complete file is included in the repository and matches the specification in the Handler section above.

### wyoming_voice_match/verify.py

See verify.py source in the main project. The complete file is included in the repository and matches the specification in the Verifier section above.

### scripts/enroll.py

See scripts/enroll.py source in the main project. The complete file is included in the repository and matches the specification in the Enrollment section above.

### scripts/test_verify.py

See scripts/test_verify.py source in the main project. The complete file is included in the repository and matches the specification in the Test Script section above.

### scripts/demo.py

See scripts/demo.py source in the main project. The complete file is included in the repository and matches the specification in the Demo Script section above.

### scripts/enroll_record.py

See scripts/enroll_record.py source in the main project. The complete file is included in the repository and matches the specification in the Satellite Enrollment Recording section above.

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
    pip uninstall -y triton 2>/dev/null; \
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
    cd /usr/local/lib/python3.10/dist-packages/torch/lib && \
    rm -f libnccl* libcublas* libcublasLt* libcusolver* \
          libcufft* libcurand* libnvrtc* libnvJitLink* libnvfuser* && \
    find /usr/local/lib/python3.10/dist-packages/torch -name "*.a" -delete && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/include && \
    rm -rf /usr/local/lib/python3.10/dist-packages/torch/share && \
    pip uninstall -y sympy networkx 2>/dev/null; \
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/recipes && \
    rm -rf /usr/local/lib/python3.10/dist-packages/speechbrain/tests && \
    find /usr/local/lib/python3.10/dist-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; \
    echo "Cleanup complete"

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

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY wyoming_voice_match/ wyoming_voice_match/
COPY scripts/ scripts/

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

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.cpu.txt .
RUN pip install --no-cache-dir \
        torch==2.4.1+cpu torchaudio==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir soundfile && \
    pip install --no-cache-dir -r requirements.cpu.txt

COPY wyoming_voice_match/ wyoming_voice_match/
COPY scripts/ scripts/

RUN mkdir -p /data/enrollment /data/voiceprints /data/models

EXPOSE 10350

ENTRYPOINT ["python", "-m", "wyoming_voice_match"]
```

### docker-compose.yml

```yaml
services:
  wyoming-voice-match:
    image: ghcr.io/jxlarrea/wyoming-voice-match:latest
    container_name: wyoming-voice-match
    restart: unless-stopped
    ports:
      - "10350:10350"
    volumes:
      - ./data:/data
    environment:
      - UPSTREAM_URI=tcp://wyoming-faster-whisper:10300
      - LISTEN_URI=tcp://0.0.0.0:10350
      - VERIFY_THRESHOLD=0.30
      - EXTRACTION_THRESHOLD=0.25
      - LOG_LEVEL=DEBUG
      - HF_HOME=/data/hf_cache
      # - REQUIRE_SPEAKER_MATCH=true       # Set to false to forward unmatched audio
      # - TAG_SPEAKER=false                # Prepend [speaker_name] to transcripts
      # - ISOLATE_VOICE=0                  # Voice isolation: 0=off, 0.5=moderate, 1.0=full
      # - MAX_VERIFY_SECONDS=5.0           # First-pass verification window
      # - VERIFY_WINDOW_SECONDS=3.0        # Sliding window size for fallback pass
      # - VERIFY_STEP_SECONDS=1.5          # Sliding window step size
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
    image: ghcr.io/jxlarrea/wyoming-voice-match:cpu
    container_name: wyoming-voice-match
    restart: unless-stopped
    ports:
      - "10350:10350"
    volumes:
      - ./data:/data
    environment:
      - UPSTREAM_URI=tcp://wyoming-faster-whisper:10300
      - LISTEN_URI=tcp://0.0.0.0:10350
      - VERIFY_THRESHOLD=0.30
      - EXTRACTION_THRESHOLD=0.25
      - LOG_LEVEL=DEBUG
      - HF_HOME=/data/hf_cache
      # - REQUIRE_SPEAKER_MATCH=true       # Set to false to forward unmatched audio
      # - TAG_SPEAKER=false                # Prepend [speaker_name] to transcripts
      # - ISOLATE_VOICE=0                  # Voice isolation: 0=off, 0.5=moderate, 1.0=full
      # - MAX_VERIFY_SECONDS=5.0           # First-pass verification window
      # - VERIFY_WINDOW_SECONDS=3.0        # Sliding window size for fallback pass
      # - VERIFY_STEP_SECONDS=1.5          # Sliding window step size
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
scipy>=1.11.0
numpy>=1.24.0
requests>=2.28.0
huggingface_hub<0.27.0
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