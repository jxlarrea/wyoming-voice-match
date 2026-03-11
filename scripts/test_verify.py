"""Test script — verify a WAV file against enrolled voiceprints.

Usage:
    python -m scripts.test_verify /path/to/test.wav [--threshold 0.45]

Useful for tuning the similarity threshold before deploying.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import torchaudio
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Test speaker verification against a WAV file."""
    parser = argparse.ArgumentParser(
        description="Test a WAV file against enrolled voiceprints"
    )
    parser.add_argument(
        "audio_file",
        help="Path to a WAV file to verify",
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
        "--threshold",
        type=float,
        default=float(os.environ.get("THRESHOLD", "0.45")),
        help="Similarity threshold",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        _LOGGER.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    voiceprints_dir = Path(args.voiceprints_dir)
    if not voiceprints_dir.exists():
        _LOGGER.error("Voiceprints directory not found: %s", voiceprints_dir)
        sys.exit(1)

    # Load voiceprints
    voiceprints = {}
    for npy_file in sorted(voiceprints_dir.glob("*.npy")):
        voiceprints[npy_file.stem] = np.load(str(npy_file))

    if not voiceprints:
        _LOGGER.error("No voiceprints found in %s", voiceprints_dir)
        sys.exit(1)

    # Load model
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    run_opts = {"device": device} if device == "cuda" else {}
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=f"{args.model_dir}/spkrec-ecapa-voxceleb",
        run_opts=run_opts,
    )

    # Load and process audio
    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    signal = torch.from_numpy(data).unsqueeze(0)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        signal = resampler(signal)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    duration = signal.shape[1] / 16000

    # Extract embedding
    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    emb = embedding.squeeze().cpu().numpy()

    # Compare against all voiceprints
    best_score = -1.0
    best_speaker = None

    print()
    print(f"  Audio file:  {audio_path.name}")
    print(f"  Duration:    {duration:.1f}s")
    print(f"  Threshold:   {args.threshold:.2f}")
    print()
    print(f"  {'Speaker':<20} {'Similarity':>10}   Result")
    print(f"  {'─' * 20} {'─' * 10}   {'─' * 10}")

    for name, voiceprint in sorted(voiceprints.items()):
        similarity = 1.0 - cosine(emb, voiceprint)
        is_match = similarity >= args.threshold
        marker = "✓ MATCH" if is_match else "✗"
        print(f"  {name:<20} {similarity:>10.4f}   {marker}")

        if similarity > best_score:
            best_score = similarity
            best_speaker = name

    overall_match = best_score >= args.threshold
    print()
    print(
        f"  Best match:  {best_speaker} ({best_score:.4f}) → "
        f"{'✓ ACCEPTED' if overall_match else '✗ REJECTED'}"
    )
    print()


if __name__ == "__main__":
    main()
