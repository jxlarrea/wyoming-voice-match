"""Enrollment script — generate a voiceprint from WAV samples.

Usage:
    python -m scripts.enroll --speaker juan [--enrollment-dir /data/enrollment]
    python -m scripts.enroll --speaker maria
    python -m scripts.enroll --list

Place 16kHz mono WAV files in data/enrollment/<speaker_name>/, then run this
script to compute an averaged speaker embedding (voiceprint).
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
from speechbrain.inference.speaker import EncoderClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def main() -> None:
    """Run enrollment to generate a voiceprint."""
    parser = argparse.ArgumentParser(
        description="Enroll speaker voice samples to create a voiceprint"
    )
    parser.add_argument(
        "--speaker",
        help="Name of the speaker to enroll (creates data/enrollment/<name>/ and data/voiceprints/<name>.npy)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all enrolled speakers",
    )
    parser.add_argument(
        "--delete",
        metavar="NAME",
        help="Delete an enrolled speaker's voiceprint",
    )
    parser.add_argument(
        "--enrollment-dir",
        default=os.environ.get("ENROLLMENT_DIR", "/data/enrollment"),
        help="Root directory containing speaker subdirectories with WAV files",
    )
    parser.add_argument(
        "--voiceprints-dir",
        default=os.environ.get("VOICEPRINTS_DIR", "/data/voiceprints"),
        help="Output directory for voiceprint .npy files",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/data/models"),
        help="Directory to cache the speaker model",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    args = parser.parse_args()

    voiceprints_dir = Path(args.voiceprints_dir)
    enrollment_dir = Path(args.enrollment_dir)

    # Handle --list
    if args.list:
        voiceprints_dir.mkdir(parents=True, exist_ok=True)
        speakers = sorted(f.stem for f in voiceprints_dir.glob("*.npy"))
        if speakers:
            print(f"\nEnrolled speakers ({len(speakers)}):")
            for name in speakers:
                print(f"  • {name}")
            print()
        else:
            print("\nNo speakers enrolled yet.")
            print(f"Run: python -m scripts.enroll --speaker <name>\n")
        return

    # Handle --delete
    if args.delete:
        vp_file = voiceprints_dir / f"{args.delete}.npy"
        if vp_file.exists():
            vp_file.unlink()
            _LOGGER.info("Deleted voiceprint for '%s'", args.delete)
        else:
            _LOGGER.error("No voiceprint found for '%s'", args.delete)
            sys.exit(1)
        return

    # Handle --speaker (enrollment)
    if not args.speaker:
        parser.print_help()
        print("\nError: --speaker, --list, or --delete is required.")
        sys.exit(1)

    speaker_name = args.speaker.strip().lower()
    speaker_dir = enrollment_dir / speaker_name

    if not speaker_dir.exists():
        speaker_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info(
            "Created enrollment directory: %s", speaker_dir
        )
        _LOGGER.info(
            "Place WAV files (16kHz, mono, 3-10s each) in this directory, "
            "then run this command again."
        )
        return

    # Find audio files
    audio_files = sorted(
        f
        for f in speaker_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        _LOGGER.error(
            "No audio files found in %s. "
            "Place WAV files (16kHz, mono) in this directory and try again.",
            speaker_dir,
        )
        sys.exit(1)

    _LOGGER.info(
        "Enrolling '%s' — found %d audio file(s) in %s",
        speaker_name,
        len(audio_files),
        speaker_dir,
    )

    # Load model
    _LOGGER.info("Loading ECAPA-TDNN model...")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        _LOGGER.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    run_opts = {"device": device} if device == "cuda" else {}
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=f"{args.model_dir}/spkrec-ecapa-voxceleb",
        run_opts=run_opts,
    )

    # Extract embeddings from each sample
    embeddings = []
    for audio_file in audio_files:
        _LOGGER.info("Processing: %s", audio_file.name)
        try:
            data, sample_rate = sf.read(str(audio_file), dtype="float32")
            signal = torch.from_numpy(data).unsqueeze(0)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                _LOGGER.info(
                    "  Resampling from %d Hz to 16000 Hz", sample_rate
                )
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                signal = resampler(signal)

            # Convert to mono if needed
            if signal.shape[0] > 1:
                _LOGGER.info("  Converting to mono")
                signal = signal.mean(dim=0, keepdim=True)

            duration = signal.shape[1] / 16000
            _LOGGER.info("  Duration: %.1f seconds", duration)

            if duration < 1.0:
                _LOGGER.warning(
                    "  Skipping — audio is too short (< 1 second)"
                )
                continue

            with torch.no_grad():
                embedding = classifier.encode_batch(signal)

            emb_np = embedding.squeeze().cpu().numpy()
            embeddings.append(emb_np)
            _LOGGER.info("  Embedding extracted (shape=%s)", emb_np.shape)

        except Exception:
            _LOGGER.exception("  Failed to process %s", audio_file.name)
            continue

    if not embeddings:
        _LOGGER.error("No valid embeddings extracted. Check your audio files.")
        sys.exit(1)

    # Average all embeddings to create the voiceprint
    voiceprint = np.mean(embeddings, axis=0)

    # Normalize the voiceprint (unit vector for cosine similarity)
    norm = np.linalg.norm(voiceprint)
    if norm > 0:
        voiceprint = voiceprint / norm

    # Save
    voiceprints_dir.mkdir(parents=True, exist_ok=True)
    output_path = voiceprints_dir / f"{speaker_name}.npy"
    np.save(str(output_path), voiceprint)

    _LOGGER.info(
        "Voiceprint for '%s' saved to %s (from %d sample(s))",
        speaker_name,
        output_path,
        len(embeddings),
    )


if __name__ == "__main__":
    main()
