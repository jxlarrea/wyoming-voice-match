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


class SpeakerVerifyHandler(AsyncEventHandler):
    """Wyoming ASR handler that gates transcription on speaker identity.

    Runs speaker verification early (as soon as enough audio is buffered).
    Once verified, waits for the full audio stream, then uses voiceprint-based
    speaker extraction to isolate the enrolled speaker's voice from background
    noise (TV, radio, other people) before forwarding to ASR.
    """

    def __init__(
        self,
        wyoming_info: Info,
        verifier: SpeakerVerifier,
        upstream_uri: str,
        tag_speaker: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info = wyoming_info
        self.verifier = verifier
        self.upstream_uri = upstream_uri
        self.tag_speaker = tag_speaker

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
        self._audio_stopped = asyncio.Event()

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
            self._audio_stopped = asyncio.Event()
            self._stream_start_time = time.monotonic()
            _LOGGER.debug("[%s] ── New audio session started ──", sid)
            return True

        # Audio data — accumulate and trigger early verification + ASR
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._audio_buffer += chunk.audio

            # Trigger verification once we have enough audio
            if not self._verify_started and not self._responded:
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
            self._audio_stopped.set()
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
            verify_start = time.monotonic()
            result = await loop.run_in_executor(
                None,
                self.verifier.verify,
                verify_audio,
                self._audio_rate,
            )
            verify_ms = (time.monotonic() - verify_start) * 1000

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

        # Mark as responded immediately so AudioStop handler doesn't
        # trigger _process_audio_sync while we're waiting for the command
        self._responded = True

        bytes_per_second = (
            self._audio_rate * self._audio_width * self._audio_channels
        )

        # Wait until the satellite sends AudioStop (stream fully ended).
        # This ensures we capture the entire command regardless of length.
        # Cap at 30s as absolute safety.
        _LOGGER.debug("[%s] Waiting for AudioStop", sid)
        try:
            await asyncio.wait_for(self._audio_stopped.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            _LOGGER.debug("[%s] AudioStop timeout (30s)", sid)

        buffer_duration = len(self._audio_buffer) / bytes_per_second
        _LOGGER.debug("[%s] Stream ended: %.1fs buffer", sid, buffer_duration)

        # Extract only the verified speaker's audio segments using
        # voiceprint comparison. This removes TV/radio/other speakers
        # by keeping only segments that match the enrolled voice.
        full_buffer = bytes(self._audio_buffer)

        speaker_name = result.matched_speaker
        extract_ms = 0.0
        if speaker_name:
            _LOGGER.debug(
                "[%s] Extracting speaker '%s' audio from %.1fs buffer",
                sid, speaker_name, buffer_duration,
            )
            async with _MODEL_LOCK:
                loop = asyncio.get_running_loop()
                extract_start = time.monotonic()
                forward_audio = await loop.run_in_executor(
                    None,
                    self.verifier.extract_speaker_audio,
                    full_buffer,
                    speaker_name,
                    self._audio_rate,
                )
                extract_ms = (time.monotonic() - extract_start) * 1000
        else:
            forward_audio = full_buffer

        asr_duration = len(forward_audio) / bytes_per_second
        _LOGGER.debug("[%s] Forwarding %.1fs to ASR", sid, asr_duration)

        # Forward to ASR and respond immediately
        transcript = await self._forward_to_upstream(forward_audio)
        tagged = self._tag_transcript(transcript, speaker_name)
        await self.write_event(Transcript(text=tagged).event())
        self._responded = True

        total_elapsed = self._elapsed_ms()
        _LOGGER.info(
            "[%s] Pipeline complete in %.0fms: \"%s\" "
            "(verify=%.0fms, extract=%.0fms)",
            sid, total_elapsed, tagged,
            verify_ms, extract_ms,
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
                verify_start = time.monotonic()
                result = await loop.run_in_executor(
                    None,
                    self.verifier.verify,
                    audio_bytes,
                    self._audio_rate,
                )
                verify_ms = (time.monotonic() - verify_start) * 1000
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
                verify_start = time.monotonic()
                result = await loop.run_in_executor(
                    None,
                    self.verifier.verify,
                    audio_bytes,
                    self._audio_rate,
                )
                verify_ms = (time.monotonic() - verify_start) * 1000

        if result.is_match:
            _LOGGER.info(
                "[%s] Speaker verified: %s (similarity=%.4f, threshold=%.2f), "
                "forwarding to ASR",
                sid, result.matched_speaker, result.similarity, result.threshold,
            )
            if _LOGGER.isEnabledFor(logging.DEBUG):
                for name, score in result.all_scores.items():
                    _LOGGER.debug("[%s]   %s: %.4f", sid, name, score)

            # Forward full buffer to ASR (AudioStop path = quiet room, no trimming needed)
            _LOGGER.debug(
                "[%s] Forwarding %.1fs to ASR",
                sid, audio_duration,
            )

            asr_start = time.monotonic()
            transcript = await self._forward_to_upstream(audio_bytes)
            tagged = self._tag_transcript(transcript, result.matched_speaker)
            await self.write_event(Transcript(text=tagged).event())
            self._responded = True
            total_elapsed = self._elapsed_ms()
            _LOGGER.info(
                "[%s] Pipeline complete in %.0fms: \"%s\" "
                "(verify=%.0fms)",
                sid, total_elapsed, tagged,
                verify_ms,
            )
        else:
            total_elapsed = self._elapsed_ms()
            _LOGGER.warning(
                "[%s] Speaker rejected in %.0fms (verify=%.0fms, best=%.4f, threshold=%.2f, scores=%s)",
                sid, total_elapsed, verify_ms, result.similarity, result.threshold,
                {n: f"{s:.4f}" for n, s in result.all_scores.items()},
            )
            await self.write_event(Transcript(text="").event())
            self._responded = True

    def _tag_transcript(self, transcript: str, speaker_name: Optional[str]) -> str:
        """Prepend [speaker_name] to transcript if tagging is enabled."""
        if self.tag_speaker and speaker_name and transcript:
            return f"[{speaker_name}] {transcript}"
        return transcript

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