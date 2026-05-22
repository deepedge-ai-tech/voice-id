#!/usr/bin/env python3
"""Extract speaker embeddings from VoxCeleb1 split zip part.

Directly parses zip local file headers sequentially (no EOCD needed).
For each unique speaker, extracts one WAV, runs through VBLINKF model,
and saves as (N, 256) cohort embeddings.

Usage:
    python scripts/extract_cohort_from_part.py
    python scripts/extract_cohort_from_part.py --max-speakers 300
    python scripts/extract_cohort_from_part.py --max-speakers 300 --force
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import struct
import sys
import tempfile
from pathlib import Path

import zlib

import numpy as np

logger = logging.getLogger("extract_cohort")

# --------------------------------------------------------------------------- #
#  Vendored wespeaker loading (identical to WespeakerDeep)
# --------------------------------------------------------------------------- #

_MODEL: object | None = None


def _load_model() -> object:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if "wespeaker" not in sys.modules:
        wespeaker_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "wespeaker_deep_edge"
            / "_wespeaker"
            / "wespeaker"
        )
        init_file = wespeaker_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "wespeaker",
            str(init_file),
            submodule_search_locations=[str(wespeaker_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load vendored wespeaker: {wespeaker_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["wespeaker"] = module
        spec.loader.exec_module(module)

    import wespeaker  # type: ignore[import-untyped]

    model_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "wespeaker_deep_edge"
        / "_models"
        / "vblinkf"
    )
    logger.info("Loading VBLINKF model from %s", model_dir)
    _MODEL = wespeaker.load_model(str(model_dir), dtype="float32")
    return _MODEL


# --------------------------------------------------------------------------- #
#  Sequential zip parser (no EOCD required)
# --------------------------------------------------------------------------- #

ZIP_LOCAL_HEADER_FMT = "<IHHHHHIIIHH"  # 30 bytes total
ZIP_LOCAL_HEADER_SIZE = 30


def iter_zip_local_entries(path: str):
    """Iterate zip entries by following local file headers sequentially.

    Unlike :func:`zipfile.ZipFile`, this does not require a central
    directory (EOCD).  It reads local file headers one after another,
    jumping forward by the correct amount for each entry.

    Yields:
        ``(filename, compressed_data, compression_method)`` tuples.
    """
    with open(path, "rb") as f:
        while True:
            sig = f.read(4)
            if not sig:
                break
            if sig != b"PK\x03\x04":
                # Not a local file header — skip byte and retry
                continue

            buf = f.read(ZIP_LOCAL_HEADER_SIZE - 4)
            if len(buf) < ZIP_LOCAL_HEADER_SIZE - 4:
                break

            (
                _sig, version, flags, method, mod_time, mod_date,
                crc32, comp_size, uncomp_size,
                filename_len, extra_len,
            ) = struct.unpack(ZIP_LOCAL_HEADER_FMT, sig + buf)

            filename_bytes = f.read(filename_len)
            if len(filename_bytes) < filename_len:
                break
            filename = filename_bytes.decode("utf-8", errors="replace")

            # Skip extra field
            if extra_len > 0:
                f.read(extra_len)

            # Read compressed data
            if comp_size > 0:
                data = f.read(comp_size)
                if len(data) < comp_size:
                    break
            else:
                data = b""

            # Decompress deflated data so callers always get raw WAV bytes
            if method == 8 and data:
                data = zlib.decompress(data, -zlib.MAX_WBITS)

            yield filename, data


# --------------------------------------------------------------------------- #
#  Embedding extraction
# --------------------------------------------------------------------------- #


def _extract_embedding_from_wav(
    model: object, wav_data: bytes
) -> np.ndarray | None:
    """Extract a 256-dim speaker embedding from WAV bytes.

    Writes to a temporary file (required by wespeaker's API), runs
    the model, then cleans up.

    Args:
        model: Loaded wespeaker ``SpeakerModel``.
        wav_data: Raw WAV file bytes.

    Returns:
        256-dim float32 embedding, or ``None`` if no speech detected.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(wav_data)
        emb_tensor = model.extract_embedding(tmp_path)
        if emb_tensor is None:
            return None
        return emb_tensor.cpu().numpy().astype(np.float32)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cohort embeddings from VoxCeleb1 split zip part.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="asset/cohort/download/vox1_dev_wav_partaa",
        help="Path to split zip part file.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=300,
        help="Maximum number of speakers to extract.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="asset/cohort/cohort_embeddings.npy",
        help="Output .npy path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan zip and count speakers without extracting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    json_path = output_path.with_suffix(".json")
    max_speakers = args.max_speakers
    dry_run = args.dry_run

    if not input_path.is_file():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    if not dry_run and output_path.exists() and not args.force:
        logger.error("Output %s exists. Use --force to overwrite.", output_path)
        sys.exit(1)

    gb = input_path.stat().st_size / (1024**3)
    logger.info("Input: %s (%.1f GB)", input_path, gb)

    # ---- Dry-run: just scan ----
    if dry_run:
        speakers: set[str] = set()
        wav_count = 0
        for filename, data in iter_zip_local_entries(str(input_path)):
            if filename.startswith("wav/") and filename.endswith(".wav"):
                wav_count += 1
                speaker_id = filename.split("/")[1] if "/" in filename else ""
                speakers.add(speaker_id)
        logger.info(
            "Found %d WAV entries, %d unique speakers",
            wav_count, len(speakers),
        )
        logger.info("Would extract up to %d embeddings → %s", min(max_speakers, len(speakers)), output_path)
        return

    # ---- Real run ----
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading VBLINKF model ...")
    model = _load_model()
    logger.info("Model loaded.")

    # Process entries
    embeddings: list[np.ndarray] = []
    speaker_names: list[str] = []
    seen_speakers: set[str] = set()
    total_wavs = 0
    skipped_speaker = 0
    skipped_no_speech = 0

    for filename, data in iter_zip_local_entries(str(input_path)):
        if not filename.startswith("wav/") or not filename.endswith(".wav"):
            continue

        total_wavs += 1

        parts = filename.split("/")
        if len(parts) < 3:
            continue
        speaker_id = parts[1]

        if speaker_id in seen_speakers:
            skipped_speaker += 1
            continue
        seen_speakers.add(speaker_id)

        emb = _extract_embedding_from_wav(model, data)
        if emb is None:
            skipped_no_speech += 1
            seen_speakers.discard(speaker_id)
            continue

        embeddings.append(emb)
        speaker_names.append(speaker_id)

        if len(embeddings) % 50 == 0:
            logger.info("  [%d/%d] speakers extracted", len(embeddings), max_speakers)

        if len(embeddings) >= max_speakers:
            break

    if not embeddings:
        logger.error("No embeddings were extracted.")
        sys.exit(1)

    emb_array = np.stack(embeddings)
    np.save(str(output_path), emb_array)

    import json

    meta = {
        "embedding_dim": 256,
        "total_speakers": len(embeddings),
        "source": str(input_path),
        "speakers": speaker_names,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Cohort saved: %s  (shape=%s, dtype=%s)",
        output_path,
        emb_array.shape,
        emb_array.dtype,
    )
    logger.info(
        "Stats: %d WAVs scanned, %d speakers extracted, "
        "%d same-speaker skipped, %d no-speech skipped",
        total_wavs, len(embeddings), skipped_speaker, skipped_no_speech,
    )


if __name__ == "__main__":
    main()
