"""Entry point for wyoming-voice-match."""

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import List

import numpy as np

from wyoming.client import AsyncClient
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncServer

from . import __version__
from .enhance import SpeechEnhancer
from .handler import SpeakerVerifyHandler
from .verify import SpeakerVerifier

_LOGGER = logging.getLogger(__name__)


async def query_upstream_languages(uri: str, timeout: float = 10.0) -> List[str]:
    """Query the upstream ASR service for its supported languages."""
    try:
        async with AsyncClient.from_uri(uri) as client:
            await client.write_event(Describe().event())
            while True:
                event = await asyncio.wait_for(client.read_event(), timeout=timeout)
                if event is None:
                    break
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    languages = []
                    for asr in info.asr:
                        for model in asr.models:
                            languages.extend(model.languages)
                    # Deduplicate while preserving order
                    seen = set()
                    unique = []
                    for lang in languages:
                        if lang not in seen:
                            seen.add(lang)
                            unique.append(lang)
                    if unique:
                        _LOGGER.info(
                            "Upstream ASR supports %d language(s): %s",
                            len(unique), ", ".join(unique),
                        )
                        return unique
    except Exception as exc:
        _LOGGER.warning("Could not query upstream ASR languages: %s", exc)
    return []


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
        default=float(os.environ.get("VERIFY_THRESHOLD", "0.30")),
        help="Cosine similarity threshold for verification (default: 0.30)",
    )
    parser.add_argument(
        "--extraction-threshold",
        type=float,
        default=float(os.environ.get("EXTRACTION_THRESHOLD", "0.25")),
        help="Cosine similarity threshold for speaker extraction (default: 0.25)",
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
    parser.add_argument(
        "--require-speaker-match",
        action="store_true",
        default=os.environ.get("REQUIRE_SPEAKER_MATCH", "true").lower() in ("true", "1", "yes"),
        help="Require speaker verification to pass before forwarding audio (default: true)",
    )
    parser.add_argument(
        "--enhance-audio",
        action="store_true",
        default=os.environ.get("ENHANCE_AUDIO", "false").lower() in ("true", "1", "yes"),
        help="Run speech enhancement on extracted audio before ASR (default: false)",
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
        extraction_threshold=args.extraction_threshold,
        max_verify_seconds=args.max_verify_seconds,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    if not verifier.voiceprints:
        if args.require_speaker_match:
            _LOGGER.error(
                "No voiceprints found in %s. "
                "Run the enrollment script first: "
                "python -m scripts.enroll --speaker <n>",
                voiceprints_dir,
            )
            sys.exit(1)
        else:
            _LOGGER.warning(
                "No voiceprints found in %s — running in bypass mode "
                "(all audio forwarded without verification)",
                voiceprints_dir,
            )

    _LOGGER.info(
        "Speaker verifier ready — %d speaker(s) enrolled "
        "(threshold=%.2f, extraction=%.2f, device=%s, verify_window=%.1fs, "
        "sliding_window=%.1fs/%.1fs, require_match=%s)",
        len(verifier.voiceprints),
        args.threshold,
        args.extraction_threshold,
        args.device,
        args.max_verify_seconds,
        args.window_seconds,
        args.step_seconds,
        args.require_speaker_match,
    )

    # Load speech enhancer (optional)
    enhancer = None
    if args.enhance_audio:
        enhancer = SpeechEnhancer(
            model_dir=args.model_dir,
            device=args.device,
        )
    else:
        _LOGGER.debug("Speech enhancement disabled (ENHANCE_AUDIO=false)")

    # Build Wyoming service info
    # Query upstream ASR for supported languages so HA can assign
    # this proxy to any pipeline the upstream supports
    upstream_languages = await query_upstream_languages(args.upstream_uri)
    if not upstream_languages:
        _LOGGER.warning(
            "Could not detect upstream ASR languages, defaulting to all. "
            "Ensure the upstream ASR is running at %s", args.upstream_uri,
        )

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
                        languages=upstream_languages,
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
            args.require_speaker_match,
            enhancer,
        )
    )


def run() -> None:
    """Sync wrapper for main."""
    asyncio.run(main())


if __name__ == "__main__":
    run()