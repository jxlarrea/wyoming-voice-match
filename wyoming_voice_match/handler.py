"""Wyoming event handler for speaker-verified ASR proxy."""

import asyncio
import logging
import time
import uuid
from typing import Optional

import numpy as np

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
        asr_max_seconds: float = 8.0,
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

        # Use the live buffer (not the verification snapshot). After
        # verification passes, wait for the user to stop speaking before
        # forwarding. We detect this by monitoring the RMS energy of the
        # most recent audio — when it drops below the speech level, the
        # user has finished and only background noise (or silence) remains.
        bytes_per_second = (
            self._audio_rate * self._audio_width * self._audio_channels
        )
        target_bytes = int(self.asr_max_seconds * bytes_per_second)

        # Get the speech peak energy from verification to calibrate
        # what "user stopped speaking" looks like
        speech_peak = self._get_speech_peak(result, bytes_per_second)

        if speech_peak and len(self._audio_buffer) < target_bytes:
            # Speech was detected — wait for the user to finish speaking
            # "Finished" = recent audio energy dropped well below speech peak
            stop_threshold = speech_peak * 0.10  # 10% of speech peak
            check_window = int(0.5 * bytes_per_second)  # check last 500ms
            min_quiet_checks = 3  # need 3 consecutive quiet checks (300ms)
            quiet_count = 0

            wait_needed = (target_bytes - len(self._audio_buffer)) / bytes_per_second
            _LOGGER.debug(
                "[%s] Waiting up to %.1fs for speech to end "
                "(peak=%.0f, stop_threshold=%.0f)",
                sid, wait_needed, speech_peak, stop_threshold,
            )
            poll_interval = 0.1
            waited = 0.0

            while len(self._audio_buffer) < target_bytes and waited < wait_needed:
                await asyncio.sleep(poll_interval)
                waited += poll_interval

                # Check if recent audio has gone quiet
                if len(self._audio_buffer) >= check_window:
                    tail = self._audio_buffer[-check_window:]
                    tail_samples = np.frombuffer(tail, dtype=np.int16).astype(np.float32)
                    tail_rms = float(np.sqrt(np.mean(tail_samples ** 2)))

                    if tail_rms < stop_threshold:
                        quiet_count += 1
                        if quiet_count >= min_quiet_checks:
                            _LOGGER.debug(
                                "[%s] Speech ended (tail RMS=%.0f < %.0f), "
                                "forwarding after %.1fs wait",
                                sid, tail_rms, stop_threshold, waited,
                            )
                            break
                    else:
                        quiet_count = 0  # reset — still speaking or loud TV

            if waited >= wait_needed:
                _LOGGER.debug(
                    "[%s] Buffer reached %.1fs limit after %.1fs wait",
                    sid, self.asr_max_seconds, waited,
                )
        elif len(self._audio_buffer) < target_bytes:
            # No speech peak detected — just wait for buffer to fill
            wait_needed = (target_bytes - len(self._audio_buffer)) / bytes_per_second
            _LOGGER.debug(
                "[%s] No speech peak, waiting up to %.1fs for buffer",
                sid, wait_needed,
            )
            poll_interval = 0.1
            waited = 0.0
            while len(self._audio_buffer) < target_bytes and waited < wait_needed:
                await asyncio.sleep(poll_interval)
                waited += poll_interval

        forward_audio = bytes(self._audio_buffer)
        asr_audio = self._trim_for_asr(forward_audio, result, bytes_per_second)
        audio_duration = len(forward_audio) / bytes_per_second
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

    def _get_speech_peak(
        self,
        result: VerificationResult,
        bytes_per_second: int,
    ) -> Optional[float]:
        """Get the peak RMS energy of the detected speech segment.

        Used to calibrate "end of speech" detection — when recent audio
        drops well below this level, the user has stopped talking.
        Returns None if no speech segment was detected.
        """
        if result.speech_audio is None:
            return None

        samples = np.frombuffer(result.speech_audio, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return None

        # Compute RMS in 50ms frames and return the peak
        frame_size = int(self._audio_rate * 0.05)
        num_frames = len(samples) // frame_size
        if num_frames == 0:
            return float(np.sqrt(np.mean(samples ** 2)))

        frames = samples[:num_frames * frame_size].reshape(num_frames, frame_size)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        return float(np.max(rms))

    # RMS energy above this level in the tail indicates background noise (TV, radio)
    _NOISE_FLOOR_RMS = 500.0
    # How many seconds of the tail to sample for noise detection
    _NOISE_TAIL_SECONDS = 1.0

    def _trim_for_asr(
        self,
        audio_bytes: bytes,
        result: VerificationResult,
        bytes_per_second: int,
    ) -> bytes:
        """Trim audio for ASR, but only if background noise is detected.

        Checks the RMS energy of the last second of audio. If it's
        near-silence, sends the full buffer (no trimming needed). If
        there's sustained energy (TV, radio), trims to asr_max_seconds
        to cut off the noise tail.
        """
        max_bytes = int(self.asr_max_seconds * bytes_per_second)
        audio_duration = len(audio_bytes) / bytes_per_second

        # If buffer is shorter than the cap, no trimming needed regardless
        if len(audio_bytes) <= max_bytes:
            _LOGGER.debug(
                "[%s] ASR trim: buffer (%.1fs) within limit (%.1fs), sending all",
                self._session_id, audio_duration, self.asr_max_seconds,
            )
            return audio_bytes

        # Check the tail of the buffer for background noise
        tail_bytes = int(self._NOISE_TAIL_SECONDS * bytes_per_second)
        tail = audio_bytes[-tail_bytes:]
        tail_samples = np.frombuffer(tail, dtype=np.int16).astype(np.float32)
        tail_rms = float(np.sqrt(np.mean(tail_samples ** 2)))

        if tail_rms < self._NOISE_FLOOR_RMS:
            # Tail is quiet — no background noise, send full buffer
            _LOGGER.debug(
                "[%s] ASR trim: tail RMS=%.0f (quiet), sending full %.1fs",
                self._session_id, tail_rms, audio_duration,
            )
            return audio_bytes
        else:
            # Tail has sustained energy — background noise, trim
            trimmed = audio_bytes[:max_bytes]
            _LOGGER.debug(
                "[%s] ASR trim: tail RMS=%.0f (noisy), trimming to %.1fs of %.1fs",
                self._session_id, tail_rms,
                len(trimmed) / bytes_per_second, audio_duration,
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