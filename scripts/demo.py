"""Demo script - run the full verification, extraction, and enhancement pipeline on a WAV file.

Simulates what happens when audio is processed by the proxy: verifies the
speaker, extracts their voice segments, runs speech enhancement, and writes
the results as WAV files you can listen to and compare against the original.

Thresholds are read from VERIFY_THRESHOLD and EXTRACTION_THRESHOLD environment
variables (set in docker-compose.yml), matching the main service configuration.

Usage:
    python -m scripts.demo --speaker john --input test.wav --output cleaned.wav

Produces two output files:
    cleaned.wav           - extracted speaker audio (what ASR normally receives)
    cleaned_enhanced.wav  - same audio after SepFormer speech enhancement
"""

import argparse
import logging
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from wyoming_voice_match.verify import SpeakerVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
    """Write raw 16-bit mono PCM bytes as a WAV file."""
    num_channels = 1
    sample_width = 2  # 16-bit
    byte_rate = sample_rate * num_channels * sample_width
    block_align = num_channels * sample_width
    data_size = len(pcm_bytes)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM format
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", sample_width * 8))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_bytes)


def main() -> None:
    """Run the full verification and extraction pipeline on a WAV file."""
    parser = argparse.ArgumentParser(
        description="Demo: verify and extract speaker audio from a WAV file"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input WAV file (can contain voice + TV/background noise)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output WAV file (will contain only the extracted speaker audio)",
    )
    parser.add_argument(
        "--speaker", "-s",
        required=True,
        help="Name of the enrolled speaker to extract",
    )
    parser.add_argument(
        "--voiceprints-dir",
        default=os.environ.get("VOICEPRINTS_DIR", "/data/voiceprints"),
        help="Directory containing voiceprint .npy files",
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

    threshold = float(os.environ.get("VERIFY_THRESHOLD", "0.30"))
    extraction_threshold = float(os.environ.get("EXTRACTION_THRESHOLD", "0.25"))

    logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n  Error: Input file not found: {input_path}\n")
        sys.exit(1)

    # Load and prepare audio
    print(f"\n  Loading audio: {input_path}")
    signal, sample_rate = torchaudio.load(str(input_path))

    if sample_rate != 16000:
        print(f"  Resampling: {sample_rate} Hz -> 16000 Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        signal = resampler(signal)
        sample_rate = 16000

    if signal.shape[0] > 1:
        print(f"  Converting: stereo -> mono")
        signal = signal.mean(dim=0, keepdim=True)

    # Convert to raw 16-bit PCM bytes (what the proxy works with)
    audio_np = (signal.squeeze().numpy() * 32768.0).astype(np.int16)
    audio_bytes = audio_np.tobytes()
    input_duration = len(audio_bytes) / (sample_rate * 2)

    print(f"  Duration:    {input_duration:.1f}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  PCM size:    {len(audio_bytes):,} bytes")

    # Initialize verifier
    print(f"\n  Loading speaker model...")
    verifier = SpeakerVerifier(
        voiceprints_dir=args.voiceprints_dir,
        model_dir=args.model_dir,
        device=args.device,
        threshold=threshold,
        extraction_threshold=extraction_threshold,
    )

    if not verifier.voiceprints:
        print(f"\n  Error: No voiceprints found in {args.voiceprints_dir}\n")
        sys.exit(1)

    if args.speaker not in verifier.voiceprints:
        available = ", ".join(sorted(verifier.voiceprints.keys()))
        print(f"\n  Error: Speaker '{args.speaker}' not enrolled")
        print(f"  Available speakers: {available}\n")
        sys.exit(1)

    print(f"  Device:      {verifier.device}")
    print(f"  Threshold:   {threshold:.2f} (extraction: {extraction_threshold:.2f})")
    print(f"  Speaker:     {args.speaker}")

    # Step 1: Verification
    print(f"\n  --- Speaker Verification ---")
    verify_start = time.monotonic()
    result = verifier.verify(audio_bytes, sample_rate)
    verify_ms = (time.monotonic() - verify_start) * 1000

    print(f"\n  {'Speaker':<20} {'Similarity':>10}   Result")
    print(f"  {'─' * 20} {'─' * 10}   {'─' * 10}")

    for name, score in sorted(result.all_scores.items()):
        is_match = score >= threshold
        marker = "MATCH" if is_match else ""
        print(f"  {name:<20} {score:>10.4f}   {marker}")

    status = "ACCEPTED" if result.is_match else "REJECTED"
    print(f"\n  Result: {status} (best={result.similarity:.4f}, took {verify_ms:.0f}ms)")

    if not result.is_match:
        print(f"\n  Speaker not verified. No output file written.")
        print(f"  Try lowering the threshold or re-enrolling with more samples.\n")
        sys.exit(0)

    # Step 2: Speaker extraction
    print(f"\n  --- Speaker Extraction ---")
    extract_start = time.monotonic()
    extracted_bytes = verifier.extract_speaker_audio(
        audio_bytes, args.speaker, sample_rate
    )
    extract_ms = (time.monotonic() - extract_start) * 1000

    output_duration = len(extracted_bytes) / (sample_rate * 2)
    removed_duration = input_duration - output_duration
    removed_pct = (removed_duration / input_duration * 100) if input_duration > 0 else 0

    print(f"\n  Input duration:    {input_duration:.1f}s")
    print(f"  Output duration:   {output_duration:.1f}s")
    print(f"  Removed:           {removed_duration:.1f}s ({removed_pct:.0f}%)")
    print(f"  Extraction time:   {extract_ms:.0f}ms")

    # Write output WAV
    output_path = Path(args.output)
    write_wav(str(output_path), extracted_bytes, sample_rate)

    total_ms = (time.monotonic() - verify_start) * 1000
    print(f"\n  Output written to: {output_path}")
    print(f"  Total pipeline:    {total_ms:.0f}ms")

    # Step 3: Speech enhancement
    print(f"\n  --- Speech Enhancement ---")
    from wyoming_voice_match.enhance import SpeechEnhancer

    enhance_start = time.monotonic()
    enhancer = SpeechEnhancer(
        model_dir=args.model_dir,
        device=args.device,
        enhance_amount=1.0,
    )
    load_ms = (time.monotonic() - enhance_start) * 1000
    print(f"  Model loaded:      {load_ms:.0f}ms")

    # Run full enhancement once at 100%
    enhancer.enhance_amount = 1.0
    infer_start = time.monotonic()
    full_enhanced = enhancer.enhance(extracted_bytes, sample_rate)
    infer_ms = (time.monotonic() - infer_start) * 1000
    print(f"  Enhancement time:  {infer_ms:.0f}ms")

    # Write full enhanced version
    stem = output_path.stem
    enhanced_path = output_path.with_name(f"{stem}_enhanced{output_path.suffix}")
    write_wav(str(enhanced_path), full_enhanced, sample_rate)
    print(f"  Enhanced (100%):   {enhanced_path}")

    # Write blended versions at 25%, 50%, 75% for comparison
    # Re-run enhancement with different amounts (uses energy-adaptive blending)
    for pct in [25, 50, 75]:
        enhancer.enhance_amount = pct / 100.0
        blended_bytes = enhancer.enhance(extracted_bytes, sample_rate)
        blend_path = output_path.with_name(f"{stem}_enhanced_{pct}{output_path.suffix}")
        write_wav(str(blend_path), blended_bytes, sample_rate)
        print(f"  Enhanced ({pct:>3d}%):   {blend_path}")

    print(f"\n  Compare files to find your preferred ENHANCE_AMOUNT:")
    print(f"    {output_path.name:<40s}  (ENHANCE_AUDIO=false)")
    print(f"    {stem}_enhanced_25{output_path.suffix:<25s}  (ENHANCE_AMOUNT=0.25)")
    print(f"    {stem}_enhanced_50{output_path.suffix:<25s}  (ENHANCE_AMOUNT=0.50)")
    print(f"    {stem}_enhanced_75{output_path.suffix:<25s}  (ENHANCE_AMOUNT=0.75)")
    print(f"    {stem}_enhanced{output_path.suffix:<28s}  (ENHANCE_AMOUNT=1.00)")
    print()


if __name__ == "__main__":
    main()