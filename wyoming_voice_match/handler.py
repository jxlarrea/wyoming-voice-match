"""Wyoming event handler for speaker-verified ASR proxy."""

import asyncio
import logging
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .verify import SpeakerVerifier

_LOGGER = logging.getLogger(__name__)

# Lock to prevent concurrent model inference
_MODEL_LOCK = asyncio.Lock()


class SpeakerVerifyHandler(AsyncEventHandler):
    """Wyoming ASR handler that gates transcription on speaker identity.

    Buffers incoming audio, verifies the speaker against the enrolled
    voiceprint, and either forwards to the upstream ASR or returns an
    empty transcript.
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

        # Audio stream start — reset buffer
        if AudioStart.is_type(event.type):
            self._audio_buffer = bytes()
            return True

        # Audio data — accumulate
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._audio_buffer += chunk.audio
            return True

        # Audio stream end — verify speaker and optionally forward
        if AudioStop.is_type(event.type):
            await self._process_audio()
            return True

        return True

    async def _process_audio(self) -> None:
        """Verify the speaker and forward audio or reject."""
        audio_bytes = self._audio_buffer
        audio_duration = len(audio_bytes) / (
            self._audio_rate * self._audio_width * self._audio_channels
        )

        if len(audio_bytes) == 0:
            _LOGGER.debug("Empty audio buffer, returning empty transcript")
            await self.write_event(Transcript(text="").event())
            return

        _LOGGER.debug(
            "Processing %.1fs of audio (%d bytes)",
            audio_duration,
            len(audio_bytes),
        )

        # Run speaker verification (with lock for thread safety)
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
            bytes_per_second = (
                self._audio_rate * self._audio_width * self._audio_channels
            )
            if result.speech_audio is not None:
                asr_audio = result.speech_audio
                _LOGGER.debug(
                    "Forwarding speech segment to ASR (%.1fs of %.1fs)",
                    len(asr_audio) / bytes_per_second,
                    len(audio_bytes) / bytes_per_second,
                )
            else:
                max_asr_bytes = int(self.verifier.max_verify_seconds * bytes_per_second)
                if len(audio_bytes) > max_asr_bytes:
                    asr_audio = audio_bytes[:max_asr_bytes]
                    _LOGGER.debug(
                        "Trimmed audio from %.1fs to %.1fs for ASR",
                        len(audio_bytes) / bytes_per_second,
                        len(asr_audio) / bytes_per_second,
                    )
                else:
                    asr_audio = audio_bytes

            transcript = await self._forward_to_upstream(asr_audio)
            await self.write_event(Transcript(text=transcript).event())
        else:
            _LOGGER.warning(
                "Speaker rejected (best=%.4f, threshold=%.2f, scores=%s)",
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