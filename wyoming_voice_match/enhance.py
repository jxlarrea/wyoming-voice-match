"""Speech enhancement using SpeechBrain SepFormer.

Optional post-processing step that removes residual background noise
(TV, radio, ambient) from extracted speaker audio before forwarding
to ASR. Language-agnostic — operates on acoustic features, not
linguistic content.

Disabled by default (ENHANCE_AUDIO=false). When enabled, runs after
speaker extraction and before ASR forwarding.

The ENHANCE_AMOUNT parameter (0.0–1.0) controls how aggressively the
enhancement is applied by blending the enhanced signal with the original:
    0.0 = original audio (no effect)
    0.5 = 50/50 blend (moderate denoising, preserves voice naturalness)
    1.0 = full SepFormer output (maximum denoising, may distort voice)
"""

import logging
import struct
from typing import Optional

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)

# Default HuggingFace model for speech enhancement
DEFAULT_ENHANCE_MODEL = "speechbrain/sepformer-wham16k-enhancement"


class SpeechEnhancer:
    """Enhances speech audio by removing background noise.

    Uses a pretrained SepFormer model to denoise audio. The model
    operates on raw waveforms and is language-agnostic — it separates
    speech patterns from noise patterns regardless of language.

    The enhance_amount parameter blends the enhanced signal with the
    original (wet/dry mix), letting you trade off noise removal against
    voice naturalness.

    Expected input: 16-bit PCM audio at 16kHz (matching Wyoming format).
    """

    def __init__(
        self,
        model_dir: str = "/data/models",
        device: str = "cuda",
        model_source: str = DEFAULT_ENHANCE_MODEL,
        enhance_amount: float = 1.0,
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
        self.enhance_amount = max(0.0, min(1.0, enhance_amount))

        # Derive a save directory name from the model source
        model_name = model_source.replace("/", "--")
        savedir = f"{model_dir}/{model_name}"

        _LOGGER.info(
            "Loading speech enhancement model: %s (device=%s, amount=%.2f)",
            model_source, device, self.enhance_amount,
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

        Uses energy-adaptive blending to preserve the original voice in
        high-energy regions (where the speaker dominates) while applying
        SepFormer denoising to low-energy regions (where noise dominates).

        The enhance_amount parameter controls the crossover: higher values
        apply enhancement more aggressively into speech regions, lower
        values limit enhancement to only the quietest parts.

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
        waveform = original.unsqueeze(0)

        # Move to model device
        waveform = waveform.to(self.device)

        # Run enhancement — returns shape [1, num_samples, num_sources]
        with torch.no_grad():
            enhanced = self.separator.separate_batch(waveform)

        # Take the first (only) source, squeeze to 1D
        enhanced_wav = enhanced[:, :, 0].squeeze(0).cpu()

        # Energy-adaptive blending: keep original voice, enhance quiet parts
        # Compute short-term energy using a sliding window (~20ms)
        window_size = int(sample_rate * 0.02)
        energy = original.unfold(0, window_size, window_size // 2).pow(2).mean(dim=-1)

        # Normalize energy to [0, 1] range
        energy_max = energy.max()
        if energy_max > 0:
            energy_norm = energy / energy_max
        else:
            energy_norm = energy

        # Upsample energy envelope back to full signal length
        blend_mask = torch.nn.functional.interpolate(
            energy_norm.unsqueeze(0).unsqueeze(0),
            size=num_samples,
            mode="linear",
            align_corners=False,
        ).squeeze()

        # Apply enhance_amount as a threshold shift:
        # amount=1.0 → enhance everything (mask=0 everywhere)
        # amount=0.5 → enhance regions below 50% energy
        # amount=0.0 → enhance nothing (mask=1 everywhere)
        # The mask represents how much of the ORIGINAL to keep
        blend_mask = torch.clamp(
            (blend_mask - (1.0 - self.enhance_amount)) / max(self.enhance_amount, 0.01),
            0.0, 1.0,
        )

        # Smooth the mask to avoid abrupt transitions (~10ms window)
        smooth_size = int(sample_rate * 0.01)
        if smooth_size > 1:
            kernel = torch.ones(smooth_size) / smooth_size
            blend_mask = torch.nn.functional.conv1d(
                blend_mask.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=smooth_size // 2,
            ).squeeze()[:num_samples]

        # blend_mask=1 → keep original, blend_mask=0 → use enhanced
        blended = blend_mask * original + (1.0 - blend_mask) * enhanced_wav

        # Clamp to valid range and convert back to 16-bit PCM bytes
        blended = torch.clamp(blended, -1.0, 1.0)
        pcm_samples = (blended * 32767.0).to(torch.int16)
        return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples.tolist())