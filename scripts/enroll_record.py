"""Record enrollment samples directly from a Wyoming satellite.

Instead of recording WAV files on a PC and transferring them, this script
listens on the Wyoming port and captures audio streamed from your satellite
after a wake word trigger. This ensures enrollment samples match the exact
microphone characteristics of your satellite hardware.

Usage:
    # Stop the main service first:
    docker compose stop wyoming-voice-match

    # Record 5 samples:
    docker compose run --rm --service-ports --entrypoint python wyoming-voice-match \
      -m scripts.enroll_record --speaker jx --samples 10

    # Restart the main service:
    docker compose start wyoming-voice-match

The script will:
1. Start a Wyoming server on the configured listen port
2. Wait for the satellite to stream audio (triggered by wake word)
3. Save each recording as a WAV file in data/enrollment/<speaker>/
4. Respond via TTS so the satellite speaks progress updates
5. After all samples are collected, automatically run enrollment
"""

import argparse
import asyncio
import logging
import os
import struct
import subprocess
import sys
from functools import partial
from pathlib import Path

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
    """Write raw 16-bit mono PCM bytes as a WAV file."""
    num_channels = 1
    sample_width = 2
    byte_rate = sample_rate * num_channels * sample_width
    block_align = num_channels * sample_width
    data_size = len(pcm_bytes)

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", sample_width * 8))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_bytes)


class EnrollRecordState:
    """Shared state across handler instances."""

    def __init__(self, speaker_name: str, target_samples: int, output_dir: Path):
        self.speaker_name = speaker_name
        self.target_samples = target_samples
        self.output_dir = output_dir
        self.samples_recorded = 0
        self.done_event = asyncio.Event()


class EnrollRecordHandler(AsyncEventHandler):
    """Wyoming handler that records audio for enrollment."""

    def __init__(
        self,
        wyoming_info: Info,
        state: EnrollRecordState,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.state = state
        self._audio_buffer = bytes()
        self._audio_rate = 16000
        self._audio_width = 2
        self._audio_channels = 1

    async def handle_event(self, event: Event) -> bool:
        """Process a single Wyoming event."""

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if Transcribe.is_type(event.type):
            return True

        if AudioStart.is_type(event.type):
            self._audio_buffer = bytes()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._audio_buffer += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            if self.state.samples_recorded >= self.state.target_samples:
                await self.write_event(Transcript(text="").event())
                return True

            bytes_per_second = (
                self._audio_rate * self._audio_width * self._audio_channels
            )
            duration = len(self._audio_buffer) / bytes_per_second

            if duration < 1.0:
                _LOGGER.warning("Recording too short (%.1fs), skipping", duration)
                await self.write_event(Transcript(text="").event())
                return True

            self.state.samples_recorded += 1
            sample_num = self.state.samples_recorded
            remaining = self.state.target_samples - sample_num

            filename = f"satellite_{sample_num:02d}.wav"
            filepath = self.state.output_dir / filename
            write_wav(str(filepath), self._audio_buffer, self._audio_rate)

            _LOGGER.info(
                "Sample %d/%d saved: %s (%.1fs)",
                sample_num, self.state.target_samples, filepath, duration,
            )

            if remaining > 0:
                _LOGGER.info("%d sample(s) remaining", remaining)

            await self.write_event(Transcript(text="").event())

            if remaining == 0:
                self.state.done_event.set()

            return True

        return True


def main() -> None:
    """Run the enrollment recording server."""
    parser = argparse.ArgumentParser(
        description="Record enrollment samples from a Wyoming satellite"
    )
    parser.add_argument(
        "--speaker", "-s",
        required=True,
        help="Name of the speaker to record samples for",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=5,
        help="Number of samples to record (default: 5)",
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("LISTEN_URI", "tcp://0.0.0.0:10350"),
        help="URI to listen on (default: tcp://0.0.0.0:10350)",
    )
    parser.add_argument(
        "--upstream-uri",
        default=os.environ.get("UPSTREAM_URI", "tcp://localhost:10300"),
        help="Upstream ASR URI to query for supported languages",
    )
    parser.add_argument(
        "--enrollment-dir",
        default=os.environ.get("ENROLLMENT_DIR", "/data/enrollment"),
        help="Root directory for enrollment samples",
    )
    parser.add_argument(
        "--voiceprints-dir",
        default=os.environ.get("VOICEPRINTS_DIR", "/data/voiceprints"),
        help="Output directory for voiceprint .npy files",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/data/models"),
        help="Directory with cached speaker model",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    args = parser.parse_args()

    speaker_name = args.speaker.strip().lower()
    output_dir = Path(args.enrollment_dir) / speaker_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  Enrollment Recording for '{speaker_name}'")
    print(f"  ─────────────────────────────────────────")
    print(f"  Samples to record: {args.samples}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Listening on:      {args.uri}")
    print()
    print(f"  Say your wake word and speak naturally for 3-10 seconds.")
    print(f"  Vary your distance, volume, and phrasing between samples.")
    print()

    asyncio.run(
        run_server(args, speaker_name, output_dir)
    )


async def _query_upstream_languages(uri: str, timeout: float = 10.0) -> list:
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
                            len(unique), ", ".join(unique[:5]),
                        )
                        return unique
    except Exception as exc:
        _LOGGER.warning("Could not query upstream ASR languages: %s", exc)
    return []


async def run_server(args, speaker_name: str, output_dir: Path) -> None:
    """Run the Wyoming server for recording."""
    # Query upstream ASR for supported languages
    languages = await _query_upstream_languages(args.upstream_uri)
    if not languages:
        _LOGGER.warning(
            "Could not query upstream ASR at %s for languages, defaulting to ['en']",
            args.upstream_uri,
        )
        languages = ["en"]

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="voice-match-enroll",
                description="Enrollment recording mode",
                attribution=Attribution(
                    name="Wyoming Voice Match",
                    url="https://github.com/jxlarrea/wyoming-voice-match",
                ),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name="enrollment",
                        description="Recording enrollment samples",
                        languages=languages,
                        attribution=Attribution(
                            name="Wyoming Voice Match",
                            url="https://github.com/jxlarrea/wyoming-voice-match",
                        ),
                        installed=True,
                        version="1.0.0",
                    )
                ],
            )
        ]
    )

    state = EnrollRecordState(speaker_name, args.samples, output_dir)

    server = AsyncServer.from_uri(args.uri)

    # Run server in background, wait for all samples
    server_task = asyncio.create_task(
        server.run(
            partial(EnrollRecordHandler, wyoming_info, state)
        )
    )

    # Wait for all samples to be recorded
    await state.done_event.wait()

    # Give a moment for the last transcript to be sent
    await asyncio.sleep(2)

    # Cancel server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    print()
    print(f"  All {args.samples} samples recorded. Running enrollment...")
    print()

    # Run enrollment
    enroll_cmd = [
        sys.executable, "-m", "scripts.enroll",
        "--speaker", speaker_name,
        "--enrollment-dir", str(args.enrollment_dir),
        "--voiceprints-dir", str(args.voiceprints_dir),
        "--model-dir", str(args.model_dir),
        "--device", args.device,
    ]
    result = subprocess.run(enroll_cmd)

    if result.returncode == 0:
        print()
        print(f"  Enrollment complete for '{speaker_name}'.")
        print(f"  Restart the main service: docker compose start wyoming-voice-match")
        print()
    else:
        print()
        print(f"  Enrollment failed. Check the logs above for errors.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()