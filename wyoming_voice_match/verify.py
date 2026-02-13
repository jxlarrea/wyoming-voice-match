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

        # --- Pass 1: energy-based speech segment ---
        pass1_start = time.monotonic()
        speech_chunk = self._extract_speech_segment(audio_bytes, sample_rate)
        if speech_chunk is not None:
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
    ) -> Optional[bytes]:
        """Extract the segment of audio with the highest energy (likely speech).

        Computes RMS energy in short frames, finds the peak region, and
        expands outward until energy drops below a fraction of the peak.
        Returns the speech segment as raw PCM bytes, or None if audio is
        too short.
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

        return segment

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