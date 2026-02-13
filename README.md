# Wyoming Voice Match

A [Wyoming protocol](https://github.com/OHF-Voice/wyoming) ASR proxy that verifies speaker identity and isolates voice commands from background noise before forwarding audio to a downstream speech-to-text service. Designed for [Home Assistant](https://www.home-assistant.io/) voice pipelines to prevent false activations from TVs, radios, and other people — and to deliver clean transcripts even in noisy environments.

## The Problem

Home Assistant voice satellites listen for a wake word, then stream audio to a speech-to-text service. But the satellite microphone picks up everything — your voice, the TV in the background, other people talking. This causes two issues:

1. **False activations**: The TV says something that triggers a command
2. **Noisy transcripts**: Your voice command gets mixed with TV dialogue, producing garbage like *"What time is it? People look at you like some kind of service freak"*

Wyoming Voice Match solves both: it verifies that the audio contains **your voice** before allowing it through, and it trims the audio buffer so only your command — not the TV — reaches the speech-to-text service.

## How It Works

Wyoming Voice Match sits between Home Assistant and your ASR (speech-to-text) service. When a wake word is detected, Home Assistant opens a connection and starts streaming audio. Here's what happens:

```
┌──────────┐     ┌─────────────────┐     ┌───────────────────────┐     ┌──────────────┐
│   Mic    │────▶│  Wake Word      │────▶│  Wyoming Voice Match  │────▶│ ASR Service  │
│ (Device) │     │  Detection      │     │                       │     │ (Transcribe) │
└──────────┘     └─────────────────┘     │  1. Buffer audio      │     └──────────────┘
                                         │  2. Detect speech     │
                                         │  3. Verify speaker    │
                                         │  4. Trim & forward    │
                                         │     or reject         │
                                         └───────────────────────┘
```

### Step by step

1. **Buffer audio** — audio streams in from Home Assistant after the wake word is detected
2. **Detect speech** — an energy analysis isolates the loudest segment of audio (your voice near the mic), separating it from background noise like a TV
3. **Verify speaker** — the speech segment is compared against your enrolled voiceprint using an [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) neural network. If it doesn't match any enrolled speaker, the pipeline is silently stopped with an empty transcript
4. **Forward to ASR** — once verified, the first few seconds of audio are forwarded to your speech-to-text service, trimming off any trailing background noise
5. **Respond immediately** — the transcript is returned to Home Assistant without waiting for the audio stream to end, bypassing VAD delays caused by background noise

**The result:**

- In a quiet room, Voice Match adds only milliseconds of overhead to your existing pipeline — verification is nearly instant
- With a TV blaring, the VAD can't detect silence and keeps the stream open for 15+ seconds. Voice Match bypasses this by starting verification at 5 seconds and responding immediately, cutting response time to ~5 seconds
- TV audio and other speakers get rejected with low similarity scores
- Your voice commands get clean transcripts without TV dialogue mixed in

## Requirements

- Docker and Docker Compose
- NVIDIA GPU (recommended) or CPU
- A downstream Wyoming-compatible ASR service (e.g., [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper), [wyoming-onnx-asr](https://github.com/tboby/wyoming-onnx-asr))

## Quick Start

### 1. Create a Project Directory

```bash
mkdir wyoming-voice-match && cd wyoming-voice-match
mkdir -p data/enrollment
```

### 2. Create `docker-compose.yml`

**GPU (recommended):**

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
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**CPU-only:**

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
```

Update `UPSTREAM_URI` to point to your ASR service.

### 3. Enroll Your Voice

Record at least 30 WAV files per speaker, 5 seconds each, speaking naturally at varied volumes and distances. Place them in `data/enrollment/<speaker>/`:

```bash
mkdir -p data/enrollment/john
```

**Linux:**

```bash
for i in $(seq 1 30); do
  echo "Sample $i — speak naturally for 5 seconds..."
  arecord -r 16000 -c 1 -f S16_LE -d 5 "data/enrollment/john/john_$(date +%Y%m%d_%H%M%S).wav"
  sleep 1
done
```

**macOS:**

```bash
for i in $(seq 1 30); do
  echo "Sample $i — speak naturally for 5 seconds..."
  sox -d -r 16000 -c 1 -b 16 "data/enrollment/john/john_$(date +%Y%m%d_%H%M%S).wav" trim 0 5
  sleep 1
done
```

> Requires [SoX](https://formulae.brew.sh/formula/sox): `brew install sox`

**Windows (PowerShell):**

Download the [recording script](https://raw.githubusercontent.com/jxlarrea/wyoming-voice-match/main/tools/record_samples.ps1) and run it — it will list your microphones, let you pick one, and guide you through recording:

```powershell
.\record_samples.ps1 -Speaker john
```

> Requires [ffmpeg](https://ffmpeg.org/download.html): `winget install ffmpeg`

Alternatively, use any voice recorder app on your phone or computer and save the files as WAV. The enrollment script handles resampling automatically, so any sample rate or channel count will work.

**Example phrases to record** (one per sample, speak naturally):

1. "Hey, turn on the living room lights and set them to fifty percent"
2. "What's the weather going to be like tomorrow morning"
3. "Set a timer for ten minutes and remind me to check the oven"
4. "Lock the front door and turn off all the lights downstairs"
5. "What's the temperature inside the house right now"

> **Tip:** The best results come from enrolling with **30 samples** at varied volumes and distances. The more variety in your samples, the more robust your voiceprint will be.

Generate the voiceprint:

```bash
docker compose run --rm wyoming-voice-match python -m scripts.enroll --speaker john
```

Repeat for additional speakers:

```bash
docker compose run --rm wyoming-voice-match python -m scripts.enroll --speaker jane
```

Manage enrolled speakers:

```bash
# List all enrolled speakers
docker compose run --rm wyoming-voice-match python -m scripts.enroll --list

# Delete a speaker
docker compose run --rm wyoming-voice-match python -m scripts.enroll --delete john
```

### 4. Start the Service

```bash
docker compose up -d
```

### 5. Configure Home Assistant

In Home Assistant, update your voice pipeline to use this service as the STT provider:

1. Go to **Settings → Devices & Services → Wyoming Protocol**
2. Add a new Wyoming integration pointing to your server's IP on port **10350**
3. In **Settings → Voice Assistants**, edit your pipeline and set the Speech-to-Text to the new Wyoming Voice Match service

## Configuration

### Environment Variables

All configuration is done in the `environment` section of `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_URI` | `tcp://localhost:10300` | Wyoming URI of your real ASR service |
| `VERIFY_THRESHOLD` | `0.20` | Cosine similarity threshold for speaker verification (0.0–1.0) |
| `LISTEN_URI` | `tcp://0.0.0.0:10350` | URI this service listens on |
| `DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
| `HF_HOME` | `/data/hf_cache` | HuggingFace cache directory for model downloads (persisted via volume) |
| `LOG_LEVEL` | `DEBUG` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MAX_VERIFY_SECONDS` | `5.0` | Seconds of audio to buffer before starting speaker verification |
| `VERIFY_WINDOW_SECONDS` | `3.0` | Sliding window size (in seconds) for the fallback verification pass |
| `VERIFY_STEP_SECONDS` | `1.5` | Step size (in seconds) between sliding windows |
| `ASR_MAX_SECONDS` | `8.0` | Max seconds of audio buffered before forwarding to ASR after verification passes. Lower for faster response, raise for longer commands |

### Tuning the Threshold

The `VERIFY_THRESHOLD` environment variable controls how strict speaker matching is. Adjust it in `docker-compose.yml` and restart:

| Value | Behavior |
|-------|----------|
| `0.20` | **Default** — lenient, good for noisy environments with TV or background audio |
| `0.30` | Moderate — good for varied voice volumes and distances |
| `0.35` | Moderate — slightly stricter, still tolerant of quiet speech |
| `0.45` | Balanced — good security with consistent mic distance |
| `0.55` | Strict — fewer false accepts, but may reject you more often |
| `0.65` | Very strict — high security, requires close mic and clear speech |

Start with debug logging enabled and observe the similarity scores:

```bash
docker compose logs -f wyoming-voice-match
```

You'll see output like:

```
INFO [971f8eb8] Speaker verified: jx (similarity=0.3787, threshold=0.20), forwarding to ASR immediately
WARNING [3a2c1b9f] Speaker rejected in 5032ms (best=0.1847, threshold=0.20, scores={'jx': '0.1847'})
```

- **Your voice** will typically score **0.25–0.75** depending on conditions
- **TV/other speakers** will typically score **0.05–0.20**
- Set the threshold in the gap between these ranges
- If you're getting rejected when speaking quietly, **lower the threshold** or **re-enroll with more samples** recorded at different volumes and distances

> **Being rejected too often?** The most effective fix is to add more enrollment samples. Record additional samples in the conditions where you're being rejected (e.g., speaking softly, further from the mic, different times of day) and re-run enrollment. More samples produce a more robust voiceprint that handles natural voice variation better.

### Noisy Environment Tuning

The default settings are already tuned for noisy environments (TV, radio, etc.). If you need to adjust further:

```yaml
    environment:
      - MAX_VERIFY_SECONDS=5.0   # Start verification after 5s (don't wait for silence)
      - ASR_MAX_SECONDS=8.0      # Wait up to 8s of audio before forwarding to ASR
      - VERIFY_THRESHOLD=0.20           # Low threshold to account for mixed audio
```

With these settings, the proxy will verify your identity and transcribe your command within ~5 seconds, regardless of whether the satellite's VAD is still listening because of background noise.

> **Note:** The satellite may continue showing a "listening" animation after the command has already been processed. This is a cosmetic issue — Home Assistant already has your transcript and is executing the command. The satellite stops listening when its own VAD finally detects silence or times out.

### Re-enrollment

To update a speaker's voiceprint, add more WAV files to `data/enrollment/<speaker>/` and re-run enrollment. The script processes all WAV files in the folder to generate an updated voiceprint.

Record additional samples using the same method as initial enrollment, then re-run:

```bash
docker compose run --rm wyoming-voice-match python -m scripts.enroll --speaker john
docker compose restart wyoming-voice-match
```

## Docker Compose Integration

If you're running other Wyoming services, you can add Voice Match to your existing `docker-compose.yml`:

```yaml
services:
  wyoming-voice-match:
    image: jxlarrea/wyoming-voice-match:latest
    container_name: wyoming-voice-match
    restart: unless-stopped
    ports:
      - "10350:10350"
    volumes:
      - ./wyoming-voice-match/data:/data
    environment:
      - UPSTREAM_URI=tcp://wyoming-faster-whisper:10300
      - VERIFY_THRESHOLD=0.20
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

For CPU-only usage, replace the image tag with `jxlarrea/wyoming-voice-match:cpu`, remove the `deploy` section, and set `DEVICE=cpu`.

## Performance

- **Speaker verification latency:** ~5–25ms on GPU, ~200–500ms on CPU
- **End-to-end pipeline:** ~5 seconds from wake word to transcript (in noisy environments, vs 15+ seconds without early verification)
- **Memory usage:** ~500MB (model + PyTorch runtime)
- **Accuracy:** ECAPA-TDNN achieves 0.69% Equal Error Rate on VoxCeleb1, state of the art for open-source speaker verification

## Limitations

- **Short commands** (under 1–2 seconds) produce less audio for verification, reducing accuracy
- **Voice changes** from illness, whispering, or shouting may lower similarity scores — enroll with varied samples to improve robustness
- **TV noise in transcripts** can occur if `ASR_MAX_SECONDS` is set too high — the default of 8 seconds works well for most commands. Increase it if longer commands are being truncated
- **Satellite listening animation** may continue after the command has been processed, since the satellite's VAD doesn't know the proxy already responded
- **Multiple users** are supported — enroll each person separately and the service accepts audio from any enrolled speaker

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) for the ECAPA-TDNN speaker verification model
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming) by the Open Home Foundation
- [Home Assistant](https://www.home-assistant.io/) voice pipeline ecosystem