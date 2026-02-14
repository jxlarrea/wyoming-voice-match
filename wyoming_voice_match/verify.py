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
        threshold: float = 0.30,
        extraction_threshold: float = 0.25,
        max_verify_seconds: float = 5.0,
        window_seconds: float = 3.0,
        step_seconds: float = 1.5,
    ) -> None:
        self.threshold = threshold
        self.extraction_threshold = extraction_threshold
        self.max_verify_seconds = max_verify_seconds
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds

        # Auto-detect device: use CUDA if available and requested, fall back to CPU
        if device == "cuda" and not torch.cuda.is_available():
            _LOGGER.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        self.device = device

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

    def extract_speaker_audio(
        self,
        audio_bytes: bytes,
        speaker_name: str,
        sample_rate: int = 16000,
        similarity_threshold: Optional[float] = None,
    ) -> bytes:
        """Extract only the segments spoken by the given speaker.

        Uses a two-stage approach:
        1. Energy analysis to find speech regions (high-energy frames)
        2. Speaker verification on each speech region to keep only
           the enrolled speaker's voice

        Args:
            audio_bytes: Raw 16-bit PCM audio
            speaker_name: Name of the enrolled speaker to extract
            sample_rate: Audio sample rate in Hz
            similarity_threshold: Min similarity to keep a region.
                                  Defaults to self.extraction_threshold

        Returns:
            Concatenated PCM audio containing only the speaker's segments
        """
        if speaker_name not in self.voiceprints:
            _LOGGER.warning("Speaker %s not enrolled, returning full audio", speaker_name)
            return audio_bytes

        voiceprint = self.voiceprints[speaker_name]
        if similarity_threshold is None:
            similarity_threshold = self.extraction_threshold

        start_time = time.monotonic()

        # Stage 1: Find speech regions using energy analysis
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        frame_ms = 50  # 50ms frames for energy analysis
        frame_size = int(sample_rate * frame_ms / 1000)
        num_frames = len(audio_np) // frame_size

        if num_frames == 0:
            return audio_bytes

        frames = audio_np[:num_frames * frame_size].reshape(num_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Determine energy threshold: use 10th percentile * 5 as speech indicator.
        # The 10th percentile captures the quietest frames (silence/pauses),
        # giving the true noise floor even when TV fills most of the buffer.
        noise_floor = float(np.percentile(frame_rms, 10))
        energy_threshold = max(noise_floor * 5.0, 500.0)

        # Find contiguous speech regions (groups of high-energy frames)
        is_speech = frame_rms >= energy_threshold
        regions: List[tuple] = []  # (start_frame, end_frame)
        in_region = False
        region_start = 0

        for i in range(num_frames):
            if is_speech[i] and not in_region:
                region_start = i
                in_region = True
            elif not is_speech[i] and in_region:
                # Allow small gaps (up to 300ms) within speech
                gap_frames = int(0.3 * 1000 / frame_ms)
                if i + gap_frames < num_frames and np.any(is_speech[i:i + gap_frames]):
                    continue  # Skip small gap
                regions.append((region_start, i))
                in_region = False

        if in_region:
            regions.append((region_start, num_frames))

        if not regions:
            _LOGGER.debug(
                "No speech regions found (noise_floor=%.0f, threshold=%.0f)",
                noise_floor, energy_threshold,
            )
            return audio_bytes

        _LOGGER.debug(
            "Found %d speech regions (noise_floor=%.0f, threshold=%.0f): %s",
            len(regions), noise_floor, energy_threshold,
            ", ".join(
                f"{s * frame_ms / 1000:.1f}-{e * frame_ms / 1000:.1f}s"
                for s, e in regions
            ),
        )

        # Stage 2: Verify each speech region against the speaker's voiceprint
        bytes_per_sample = 2
        kept_regions = []
        region_scores = []

        # Minimum region length for sub-region scanning (seconds)
        sub_scan_min_seconds = 3.0
        # Window size for sub-region scanning (seconds)
        sub_scan_window_seconds = 1.5
        sub_scan_step_seconds = 0.5

        for start_frame, end_frame in regions:
            # Ensure minimum 1s for reliable embedding
            duration_frames = end_frame - start_frame
            min_frames = int(1.0 * 1000 / frame_ms)  # 1s minimum
            if duration_frames < min_frames:
                # Expand region symmetrically to reach 1s
                expand = (min_frames - duration_frames) // 2
                start_frame = max(0, start_frame - expand)
                end_frame = min(num_frames, end_frame + expand)

            start_byte = start_frame * frame_size * bytes_per_sample
            end_byte = end_frame * frame_size * bytes_per_sample
            region_audio = audio_bytes[start_byte:end_byte]

            embedding = self._extract_embedding(region_audio, sample_rate)
            similarity = float(1.0 - cosine(embedding, voiceprint))
            region_scores.append((start_frame, end_frame, similarity))

            if similarity >= similarity_threshold:
                region_duration = (end_frame - start_frame) * frame_ms / 1000
                if region_duration >= sub_scan_min_seconds:
                    # Stage 3: Sub-region scan to trim non-speaker edges.
                    # Uses the extraction threshold directly — it's already
                    # tuned to separate speaker voice from background audio.
                    trimmed = self._trim_region(
                        audio_bytes, start_frame, end_frame,
                        frame_size, bytes_per_sample, voiceprint,
                        sample_rate, similarity_threshold,
                        sub_scan_window_seconds, sub_scan_step_seconds,
                        frame_ms,
                    )
                    if trimmed is not None:
                        kept_regions.append(trimmed)
                    else:
                        kept_regions.append(region_audio)
                else:
                    kept_regions.append(region_audio)
            elif (end_frame - start_frame) * frame_ms / 1000 >= sub_scan_min_seconds:
                # Region failed as a whole, but it's long enough that
                # the speaker's voice may be buried inside a larger blob
                # of background audio. Scan with sliding window to rescue
                # any sub-segments that match.
                _LOGGER.debug(
                    "  Region %.1f-%.1fs failed (%.2f), scanning for buried voice",
                    start_frame * frame_ms / 1000, end_frame * frame_ms / 1000,
                    similarity,
                )
                rescued = self._trim_region(
                    audio_bytes, start_frame, end_frame,
                    frame_size, bytes_per_sample, voiceprint,
                    sample_rate, self.threshold,
                    sub_scan_window_seconds, sub_scan_step_seconds,
                    frame_ms,
                )
                if rescued is not None:
                    kept_regions.append(rescued)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        _LOGGER.debug(
            "Speaker extraction: %d/%d regions kept for '%s' in %.0fms "
            "(threshold=%.2f)",
            len(kept_regions), len(regions), speaker_name,
            elapsed_ms, similarity_threshold,
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            scores_str = " ".join(
                f"{s * frame_ms / 1000:.1f}-{e * frame_ms / 1000:.1f}s="
                f"{'KEEP' if sim >= similarity_threshold else 'drop'}({sim:.2f})"
                for s, e, sim in region_scores
            )
            _LOGGER.debug("Region scores: %s", scores_str)

        if not kept_regions:
            _LOGGER.warning(
                "No regions matched speaker '%s', returning full audio",
                speaker_name,
            )
            return audio_bytes

        return b"".join(kept_regions)

    def _trim_region(
        self,
        audio_bytes: bytes,
        start_frame: int,
        end_frame: int,
        frame_size: int,
        bytes_per_sample: int,
        voiceprint: np.ndarray,
        sample_rate: int,
        similarity_threshold: float,
        window_seconds: float,
        step_seconds: float,
        frame_ms: int,
    ) -> Optional[bytes]:
        """Trim a long kept region by scanning with a sliding window.

        Scans the entire region with overlapping windows, scores each one,
        then finds the longest contiguous stretch of matching windows and
        trims to those boundaries.

        Returns None if trimming would produce no audio.
        """
        region_start_sec = start_frame * frame_ms / 1000
        region_end_sec = end_frame * frame_ms / 1000
        region_duration = region_end_sec - region_start_sec

        window_bytes = int(window_seconds * sample_rate * bytes_per_sample)
        step_bytes = int(step_seconds * sample_rate * bytes_per_sample)
        min_window_bytes = int(1.0 * sample_rate * bytes_per_sample)

        start_byte = start_frame * frame_size * bytes_per_sample
        end_byte = end_frame * frame_size * bytes_per_sample
        region_audio = audio_bytes[start_byte:end_byte]
        region_len = len(region_audio)

        if region_len < window_bytes:
            return None

        # Scan entire region and collect all window scores
        windows = []  # (pos_byte, similarity)
        pos = 0
        while pos + window_bytes <= region_len:
            chunk = region_audio[pos:pos + window_bytes]
            embedding = self._extract_embedding(chunk, sample_rate)
            sim = float(1.0 - cosine(embedding, voiceprint))
            windows.append((pos, sim))
            chunk_start_sec = region_start_sec + pos / (sample_rate * bytes_per_sample)
            chunk_end_sec = chunk_start_sec + window_seconds
            _LOGGER.debug(
                "  Trim scan %.1f-%.1fs: %.4f %s",
                chunk_start_sec, chunk_end_sec, sim,
                "SPEAKER" if sim >= similarity_threshold else "",
            )
            pos += step_bytes

        if not windows:
            return None

        # Find the longest contiguous run of matching windows
        best_run_start = -1
        best_run_len = 0
        current_start = -1
        current_len = 0

        for i, (_, sim) in enumerate(windows):
            if sim >= similarity_threshold:
                if current_start == -1:
                    current_start = i
                    current_len = 1
                else:
                    current_len += 1
                if current_len > best_run_len:
                    best_run_start = current_start
                    best_run_len = current_len
            else:
                current_start = -1
                current_len = 0

        if best_run_start == -1:
            return None

        # Trim to the boundaries of the best run
        speaker_start_byte = windows[best_run_start][0]
        last_window_pos = windows[best_run_start + best_run_len - 1][0]
        speaker_end_byte = last_window_pos + window_bytes

        trimmed = region_audio[speaker_start_byte:speaker_end_byte]
        if len(trimmed) < min_window_bytes:
            return None

        trimmed_start_sec = region_start_sec + speaker_start_byte / (sample_rate * bytes_per_sample)
        trimmed_end_sec = region_start_sec + speaker_end_byte / (sample_rate * bytes_per_sample)
        _LOGGER.debug(
            "  Trimmed %.1f-%.1fs -> %.1f-%.1fs (%.1fs -> %.1fs)",
            region_start_sec, region_end_sec,
            trimmed_start_sec, trimmed_end_sec,
            region_duration,
            trimmed_end_sec - trimmed_start_sec,
        )

        return trimmed

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