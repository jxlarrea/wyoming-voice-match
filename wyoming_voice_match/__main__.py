"""Entry point for wyoming-voice-match."""

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import SpeakerVerifyHandler
from .verify import SpeakerVerifier

_LOGGER = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        description="Wyoming ASR proxy with speaker verification"
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("LISTEN_URI", "tcp://0.0.0.0:10350"),
        help="URI to listen on (default: tcp://0.0.0.0:10350)",
    )
    parser.add_argument(
        "--upstream-uri",
        default=os.environ.get("UPSTREAM_URI", "tcp://localhost:10300"),
        help="URI of upstream ASR service (default: tcp://localhost:10300)",
    )
    parser.add_argument(
        "--voiceprints-dir",
        default=os.environ.get("VOICEPRINTS_DIR", "/data/voiceprints"),
        help="Directory containing voiceprint .npy files (default: /data/voiceprints)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("VERIFY_THRESHOLD", "0.20")),
        help="Cosine similarity threshold for verification (default: 0.20)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/data/models"),
        help="Directory to cache the speaker model (default: /data/models)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--max-verify-seconds",
        type=float,
        default=float(os.environ.get("MAX_VERIFY_SECONDS", "5.0")),
        help="Max audio duration (seconds) for first-pass verification (default: 5.0)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=float(os.environ.get("VERIFY_WINDOW_SECONDS", "3.0")),
        help="Sliding window size in seconds for fallback verification (default: 3.0)",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=float(os.environ.get("VERIFY_STEP_SECONDS", "1.5")),
        help="Sliding window step in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--tag-speaker",
        action="store_true",
        default=os.environ.get("TAG_SPEAKER", "false").lower() in ("true", "1", "yes"),
        help="Prepend [speaker_name] to transcripts (default: false)",
    )

    return parser.parse_args()


async def main() -> None:
    """Run the Wyoming voice match proxy."""
    args = get_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate voiceprints directory exists
    voiceprints_dir = Path(args.voiceprints_dir)
    if not voiceprints_dir.exists():
        _LOGGER.error(
            "Voiceprints directory not found at %s. "
            "Run the enrollment script first: "
            "python -m scripts.enroll --speaker <name>",
            voiceprints_dir,
        )
        sys.exit(1)

    # Load speaker verifier
    _LOGGER.info("Loading ECAPA-TDNN speaker verification model...")
    verifier = SpeakerVerifier(
        voiceprints_dir=str(voiceprints_dir),
        model_dir=args.model_dir,
        device=args.device,
        threshold=args.threshold,
        max_verify_seconds=args.max_verify_seconds,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    if not verifier.voiceprints:
        _LOGGER.error(
            "No voiceprints found in %s. "
            "Run the enrollment script first: "
            "python -m scripts.enroll --speaker <name>",
            voiceprints_dir,
        )
        sys.exit(1)

    _LOGGER.info(
        "Speaker verifier ready — %d speaker(s) enrolled "
        "(threshold=%.2f, device=%s, verify_window=%.1fs, "
        "sliding_window=%.1fs/%.1fs)",
        len(verifier.voiceprints),
        args.threshold,
        args.device,
        args.max_verify_seconds,
        args.window_seconds,
        args.step_seconds,
    )

    # Build Wyoming service info
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="voice-match",
                description=f"Speaker-verified ASR proxy v{__version__}",
                attribution=Attribution(
                    name="Wyoming Voice Match",
                    url="https://github.com/jxlarrea/wyoming-voice-match",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="voice-match-proxy",
                        description="ECAPA-TDNN speaker gate → upstream ASR",
                        languages=["en"],
                        attribution=Attribution(
                            name="Wyoming Voice Match",
                            url="https://github.com/jxlarrea/wyoming-voice-match",
                        ),
                        installed=True,
                        version=__version__,
                    )
                ],
            )
        ]
    )

    _LOGGER.info(
        "Starting server on %s → upstream %s",
        args.uri,
        args.upstream_uri,
    )

    server = AsyncServer.from_uri(args.uri)
    await server.run(
        partial(
            SpeakerVerifyHandler,
            wyoming_info,
            verifier,
            args.upstream_uri,
            args.tag_speaker,
        )
    )


def run() -> None:
    """Sync wrapper for main."""
    asyncio.run(main())


if __name__ == "__main__":
    run()