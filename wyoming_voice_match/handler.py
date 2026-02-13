"""Wyoming event handler for speaker-verified ASR proxy."""

import asyncio
import logging
import time
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


class SpeakerVerifyHandler(AsyncEventHandler):
    """Wyoming ASR handler that gates transcription on speaker identity.

    Runs speaker verification early (as soon as enough audio is buffered)
    rather than waiting for AudioStop. This reduces perceived latency
    when the upstream VAD keeps the stream open due to background noise.
    """

    def __init__(
        self,
        wyoming_info: Info,
        verifier: SpeakerVerifier,
        upstream_uri: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info = wyoming_info
        self.verifier = verifier
        self.upstream_uri = upstream_uri

        # Per-connection state
        self._audio_buffer = bytes()
        self._audio_rate: int = 16000
        self._audio_width: int = 2
        self._audio_channels: int = 1
        self._language: Optional[str] = None
        self._verify_task: Optional[asyncio.Task] = None
        self._verify_result: Optional[VerificationResult] = None
        self._verify_started: bool = False
        self._stream_start_time: Optional[float] = None

    async def handle_event(self, event: Event) -> bool:
        """Process a single Wyoming event.

        Returns True to keep the connection open, False to close it.
        """
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
            self._verify_result = None
            self._verify_started = False
            self._stream_start_time = time.monotonic()
            _LOGGER.debug("── New audio session started ──")
            return True

        # Audio data — accumulate and trigger early verification
        if AudioChunk.is_type(event.type):
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
                        "Early verify: %.1fs buffered, starting verification",
                        buffered_seconds,
                    )
                    # Take a snapshot of the buffer for verification
                    verify_audio = bytes(self._audio_buffer)
                    self._verify_task = asyncio.create_task(
                        self._run_verification(verify_audio)
                    )

            return True

        # Audio stream end — use early result or verify now
        if AudioStop.is_type(event.type):
            await self._process_audio()
            return True

        return True

    async def _run_verification(self, audio_bytes: bytes) -> VerificationResult:
        """Run speaker verification in background with lock."""
        async with _MODEL_LOCK:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self.verifier.verify,
                audio_bytes,
                self._audio_rate,
            )
        self._verify_result = result
        return result

    async def _process_audio(self) -> None:
        """Verify the speaker and forward audio or reject."""
        audio_bytes = self._audio_buffer
        bytes_per_second = (
            self._audio_rate * self._audio_width * self._audio_channels
        )
        audio_duration = len(audio_bytes) / bytes_per_second

        if len(audio_bytes) == 0:
            _LOGGER.debug("Empty audio buffer, returning empty transcript")
            await self.write_event(Transcript(text="").event())
            _LOGGER.debug("── Session complete (empty) ──")
            return

        stream_elapsed = 0.0
        if self._stream_start_time is not None:
            stream_elapsed = (time.monotonic() - self._stream_start_time) * 1000

        _LOGGER.debug(
            "AudioStop received: %.1fs of audio (%d bytes), "
            "stream duration: %.0fms",
            audio_duration,
            len(audio_bytes),
            stream_elapsed,
        )

        # Wait for early verification if it was started, otherwise verify now
        if self._verify_task is not None:
            _LOGGER.debug("Waiting for early verification result...")
            wait_start = time.monotonic()
            result = await self._verify_task
            wait_elapsed = (time.monotonic() - wait_start) * 1000
            _LOGGER.debug(
                "Early verification result ready (waited %.0fms)",
                wait_elapsed,
            )
        else:
            _LOGGER.debug(
                "No early verification (only %.1fs buffered), verifying now",
                audio_duration,
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
                "Speaker verified: %s (similarity=%.4f, threshold=%.2f), "
                "forwarding to ASR",
                result.matched_speaker,
                result.similarity,
                result.threshold,
            )
            if _LOGGER.isEnabledFor(logging.DEBUG):
                for name, score in result.all_scores.items():
                    _LOGGER.debug("  %s: %.4f", name, score)

            # Use the detected speech segment for ASR to avoid sending
            # background noise (e.g., TV audio) to the transcription service.
            # Fall back to trimming to max_verify_seconds if no segment detected.
            if result.speech_audio is not None:
                asr_audio = result.speech_audio
                _LOGGER.debug(
                    "Forwarding speech segment to ASR (%.1fs of %.1fs)",
                    len(asr_audio) / bytes_per_second,
                    audio_duration,
                )
            else:
                max_asr_bytes = int(self.verifier.max_verify_seconds * bytes_per_second)
                if len(audio_bytes) > max_asr_bytes:
                    asr_audio = audio_bytes[:max_asr_bytes]
                    _LOGGER.debug(
                        "Trimmed audio from %.1fs to %.1fs for ASR",
                        audio_duration,
                        len(asr_audio) / bytes_per_second,
                    )
                else:
                    asr_audio = audio_bytes

            transcript = await self._forward_to_upstream(asr_audio)
            await self.write_event(Transcript(text=transcript).event())
            total_elapsed = 0.0
            if self._stream_start_time is not None:
                total_elapsed = (time.monotonic() - self._stream_start_time) * 1000
            _LOGGER.info(
                "Pipeline complete in %.0fms: \"%s\"",
                total_elapsed,
                transcript,
            )
        else:
            total_elapsed = 0.0
            if self._stream_start_time is not None:
                total_elapsed = (time.monotonic() - self._stream_start_time) * 1000
            _LOGGER.warning(
                "Speaker rejected in %.0fms (best=%.4f, threshold=%.2f, scores=%s)",
                total_elapsed,
                result.similarity,
                result.threshold,
                {n: f"{s:.4f}" for n, s in result.all_scores.items()},
            )
            await self.write_event(Transcript(text="").event())

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
                        _LOGGER.debug("Upstream transcript: %s", transcript.text)
                        return transcript.text

        except Exception:
            _LOGGER.exception("Error communicating with upstream ASR at %s", self.upstream_uri)
            return ""