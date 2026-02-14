"""Speech enhancement using SpeechBrain SepFormer.

Optional post-processing step that isolates the speaker's voice from
residual background noise (TV, radio, ambient) in extracted audio
before forwarding to ASR. Language-agnostic — operates on acoustic
features, not linguistic content.

Controlled by ISOLATE_VOICE (0.0–1.0):
    0.0 = disabled (no enhancement, original audio passed through)
    0.5 = moderate isolation (blend of original and enhanced)
    1.0 = full isolation (maximum noise removal)

When enabled, runs after speaker extraction and before ASR forwarding.
"""

import logging
import struct

import torch

_LOGGER = logging.getLogger(__name__)

# Default HuggingFace model for speech enhancement
DEFAULT_ENHANCE_MODEL = "speechbrain/sepformer-wham16k-enhancement"


class SpeechEnhancer:
    """Isolates speech from background noise using SepFormer.

    Uses a pretrained SepFormer model to denoise audio. The model
    operates on raw waveforms and is language-agnostic — it separates
    speech patterns from noise patterns regardless of language.

    The isolate_voice parameter controls the blend between the original
    and enhanced signals, allowing fine-tuning of the noise removal vs
    voice naturalness trade-off.

    Expected input: 16-bit PCM audio at 16kHz (matching Wyoming format).
    """

    def __init__(
        self,
        model_dir: str = "/data/models",
        device: str = "cuda",
        model_source: str = DEFAULT_ENHANCE_MODEL,
        isolate_voice: float = 1.0,
    ) -> None:
        from speechbrain.inference.separation import SepformerSeparation

        # Auto-detect device
        if device == "cuda" and not torch.cuda.is_available():
            _LOGGER.warning(
                "CUDA requested for enhancer but not available, "
                "falling back to CPU"
            )
            device = "cpu"
        self.device = device
        self.isolate_voice = max(0.0, min(1.0, isolate_voice))

        # Derive a save directory name from the model source
        model_name = model_source.replace("/", "--")
        savedir = f"{model_dir}/{model_name}"

        _LOGGER.info(
            "Loading speech enhancement model: %s "
            "(device=%s, isolate_voice=%.2f)",
            model_source, device, self.isolate_voice,
        )

        run_opts = {"device": device} if device == "cuda" else {}
        self.separator = SepformerSeparation.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts=run_opts,
        )

        _LOGGER.info("Speech enhancement model ready")

    def enhance(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
    ) -> bytes:
        """Enhance audio by removing background noise.

        Blends between original and SepFormer-enhanced audio based on
        isolate_voice (0.0 = original, 1.0 = fully enhanced).

        Args:
            audio_bytes: Raw PCM audio (16-bit signed, mono).
            sample_rate: Sample rate in Hz (must be 16000).
            sample_width: Bytes per sample (must be 2 for 16-bit).

        Returns:
            Enhanced audio as raw PCM bytes, same format as input.
        """
        if not audio_bytes:
            return audio_bytes

        # Convert raw PCM bytes to float32 tensor [-1.0, 1.0]
        num_samples = len(audio_bytes) // sample_width
        samples = struct.unpack(f"<{num_samples}h", audio_bytes)
        original = torch.FloatTensor(samples) / 32768.0
        waveform = original.unsqueeze(0).to(self.device)

        # Run enhancement — returns shape [1, num_samples, num_sources]
        with torch.no_grad():
            enhanced = self.separator.separate_batch(waveform)

        # Take the first (only) source, squeeze to 1D
        enhanced_wav = enhanced[:, :, 0].squeeze(0).cpu()

        # Blend based on isolate_voice level
        if self.isolate_voice < 1.0:
            blended = (
                self.isolate_voice * enhanced_wav
                + (1.0 - self.isolate_voice) * original
            )
        else:
            blended = enhanced_wav

        # Clamp to valid range and convert back to 16-bit PCM bytes
        blended = torch.clamp(blended, -1.0, 1.0)
        pcm_samples = (blended * 32767.0).to(torch.int16)
        return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples.tolist())