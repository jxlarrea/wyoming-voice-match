# Wyoming Voice Match

A [Wyoming protocol](https://github.com/OHF-Voice/wyoming) ASR proxy that verifies speaker identity before forwarding audio to a downstream speech-to-text service. Designed for [Home Assistant](https://www.home-assistant.io/) voice pipelines to prevent false activations from TVs, radios, and other people.

## How It Works

Wyoming Voice Match sits between your wake word detector and your ASR service. When a wake word is detected and audio streams in, this service:

1. Buffers the incoming audio
2. Extracts a speaker embedding using [SpeechBrain's ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) model
3. Compares the embedding against your enrolled voiceprint using cosine similarity
4. **If the speaker matches**: forwards audio to your real ASR service and returns the transcript
5. **If the speaker doesn't match**: returns an empty transcript, silently stopping the pipeline

```
┌──────────┐     ┌─────────────────┐     ┌───────────────────────┐     ┌──────────────┐
│   Mic    │────▶│ openwakeword    │────▶│ wyoming-voice-match   │────▶│ ASR Service  │
│ (Device) │     │ (Wake Word)     │     │ (Speaker Gate)        │     │ (Transcribe) │
└──────────┘     └─────────────────┘     └───────────────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          Speaker doesn't match?
                                          → Empty transcript returned
                                          → Pipeline stops silently
```

## Requirements

- Docker and Docker Compose
- NVIDIA GPU (recommended) or CPU
- A downstream Wyoming-compatible ASR service (e.g., [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper), [wyoming-whisper](https://github.com/rhasspy/wyoming-whisper))

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jxlarrea/wyoming-voice-match.git
cd wyoming-voice-match
```

### 2. Configure

Edit `docker-compose.yml` and set your upstream ASR URI and preferences in the `environment` section:

```yaml
    environment:
      - UPSTREAM_URI=tcp://wyoming-faster-whisper:10300
      - THRESHOLD=0.45
      - LISTEN_URI=tcp://0.0.0.0:10350
      - DEVICE=cuda
      - HF_HOME=/data/hf_cache
      - LOG_LEVEL=DEBUG
```

### 3. Enroll Your Voice

Each speaker gets their own enrollment directory with audio samples. Record 5–10 WAV files per person, 3–10 seconds each, speaking naturally.

```bash
# Create enrollment directory for a speaker (first run creates the folder)
docker compose run --rm --entrypoint python voice-match -m scripts.enroll --speaker john
```

Record audio samples and place them in `data/enrollment/john/`:

**Linux:**

List available recording devices:

```bash
arecord -l
```

```bash
for i in $(seq 1 5); do
  echo "Sample $i — speak naturally for 5 seconds..."
  arecord -r 16000 -c 1 -f S16_LE -d 5 "data/enrollment/john/sample_${i}.wav"
  sleep 1
done
```

**macOS:**

List available audio devices:

```bash
sox --help-device
# Or using system_profiler:
system_profiler SPAudioDataType
```

```bash
for i in $(seq 1 5); do
  echo "Sample $i — speak naturally for 5 seconds..."
  sox -d -r 16000 -c 1 -b 16 "data/enrollment/john/sample_${i}.wav" trim 0 5
  sleep 1
done
```

> Requires [SoX](https://formulae.brew.sh/formula/sox): `brew install sox`

**Windows (PowerShell):**

Use the included recording script — it will list your microphones, let you pick one, and guide you through recording:

```powershell
.\tools\record_samples.ps1 -Speaker john
```

> Requires [ffmpeg](https://ffmpeg.org/download.html): `winget install ffmpeg`

Alternatively, use any voice recorder app on your phone or computer and save the files as WAV. The enrollment script handles resampling automatically, so any sample rate or channel count will work.

**Example phrases to record** (one per sample, speak naturally):

1. "Hey, turn on the living room lights and set them to fifty percent"
2. "What's the weather going to be like tomorrow morning"
3. "Set a timer for ten minutes and remind me to check the oven"
4. "Play some jazz music in the kitchen please"
5. "Good morning, what's on my calendar for today"
6. "Lock the front door and turn off all the lights downstairs"
7. "What's the temperature inside the house right now"

> **Tip:** Vary your distance from the mic, volume, and phrasing across samples. Use your normal everyday voice — don't whisper or shout. The more variety, the more robust your voiceprint will be.

Then run enrollment again to generate the voiceprint:

```bash
docker compose run --rm --entrypoint python voice-match -m scripts.enroll --speaker john
```

Repeat for additional speakers:

```bash
docker compose run --rm --entrypoint python voice-match -m scripts.enroll --speaker jane
```

Manage enrolled speakers:

```bash
# List all enrolled speakers
docker compose run --rm --entrypoint python voice-match -m scripts.enroll --list

# Delete a speaker
docker compose run --rm --entrypoint python voice-match -m scripts.enroll --delete john
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
| `THRESHOLD` | `0.45` | Cosine similarity threshold for speaker verification (0.0–1.0) |
| `LISTEN_URI` | `tcp://0.0.0.0:10350` | URI this service listens on |
| `DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
| `HF_HOME` | `/data/hf_cache` | HuggingFace cache directory for model downloads (persisted via volume) |
| `LOG_LEVEL` | `DEBUG` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Tuning the Threshold

Start with `--debug` logging enabled and observe the similarity scores:

```bash
docker compose logs -f voice-match
```

You'll see output like:

```
INFO: Speaker verified (similarity=0.6234, threshold=0.45), forwarding to ASR
WARNING: Speaker rejected (similarity=0.1847, threshold=0.45)
```

- **Your voice** will typically score **0.45–0.75**
- **TV/other speakers** will typically score **0.05–0.25**
- Set the threshold in the gap between these ranges
- Lower threshold = fewer false rejections but more false accepts
- Higher threshold = more secure but may reject you if you're far from the mic or speaking unusually

### Re-enrollment

To update a speaker's voiceprint, add or replace WAV files in `data/enrollment/<n>/` and re-run:

```bash
docker compose run --rm --entrypoint python voice-match -m scripts.enroll --speaker john
```

Then restart the service:

```bash
docker compose restart voice-match
```

## Docker Compose Integration

If you're running other Wyoming services, you can add this to your existing `docker-compose.yml`:

```yaml
services:
  voice-match:
    build: ./wyoming-voice-match
    container_name: wyoming-voice-match
    restart: unless-stopped
    ports:
      - "10350:10350"
    volumes:
      - ./wyoming-voice-match/data:/data
    environment:
      - UPSTREAM_URI=tcp://wyoming-faster-whisper:10300
      - THRESHOLD=0.45
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

For CPU-only usage, use `docker-compose.cpu.yml` instead, or remove the `deploy` section and set `DEVICE=cpu`.

## Architecture

The service implements the Wyoming ASR protocol (`Transcribe`, `AudioStart`, `AudioChunk`, `AudioStop`, `Transcript`) and presents itself as a standard ASR service to Home Assistant. Internally it:

1. Receives and buffers a complete audio stream
2. Runs ECAPA-TDNN inference to extract a 192-dimensional speaker embedding
3. Computes cosine similarity against all enrolled voiceprints
4. On match (any speaker): opens an async connection to the upstream ASR, replays the buffered audio, and returns the transcript
5. On rejection: returns an empty `Transcript` event

The ECAPA-TDNN model is loaded once at startup and shared across connections. A lock prevents concurrent inference on the model.

## Performance

- **Speaker verification latency:** ~30–80ms on GPU, ~200–500ms on CPU
- **Memory usage:** ~500MB (model + PyTorch runtime)
- **Accuracy:** ECAPA-TDNN achieves 0.69% Equal Error Rate on VoxCeleb1, which is state of the art for open-source speaker verification

## Limitations

- **Short commands** (under 1–2 seconds) have less audio data, reducing verification accuracy
- **Voice changes** from illness, whispering, or shouting may lower similarity scores — enroll with varied samples
- **Multiple users** are supported — enroll each person separately and the service accepts audio from any enrolled speaker
- **Adds latency** since audio must be fully buffered before verification; typically unnoticeable in practice

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) for the ECAPA-TDNN speaker verification model
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming) by the Open Home Foundation
- [Home Assistant](https://www.home-assistant.io/) voice pipeline ecosystem