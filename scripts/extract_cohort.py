#!/usr/bin/env python3
"""Extract VoxCeleb1 WAVs from downloaded snapshot and build cohort embeddings.

Processes the ``ProgramComputer/voxceleb`` dataset downloaded via
``snapshot_download`` into the HF cache.  For each speaker in the dev set,
extracts one utterance, runs it through the VBLINKF model, and saves the
resulting (N, 256) embedding matrix to ``asset/cohort/``.

Usage:
    python scripts/extract_cohort.py --voxceleb 300
    python scripts/extract_cohort.py --voxceleb 300 --output asset/cohort/cohort_embeddings.npy
    python scripts/extract_cohort.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("extract_cohort")

# --------------------------------------------------------------------------- #
#  Vendored wespeaker loading (identical to build_cohort.py)
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
#  Snapshot helpers
# --------------------------------------------------------------------------- #

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _find_snapshot() -> Path | None:
    """Find the downloaded snapshot for ``ProgramComputer/voxceleb``.

    Returns:
        Path to the snapshot directory, or ``None`` if not found.
    """
    dataset_dir = HF_CACHE / "datasets--ProgramComputer--voxceleb" / "snapshots"
    if not dataset_dir.is_dir():
        return None
    snapshots = sorted(dataset_dir.iterdir(), reverse=True)
    return snapshots[0] if snapshots else None


def _find_zip_parts(snapshot: Path) -> list[Path]:
    """Find all ``vox1_dev_wav_part*`` files in the snapshot.

    Args:
        snapshot: Snapshot directory path.

    Returns:
        Sorted list of zip part file paths.
    """
    return sorted(snapshot.glob("vox1/vox1_dev_wav_part*"))


def _find_full_zip(snapshot: Path) -> Path | None:
    """Find ``vox1_dev_wav.zip`` in the snapshot.

    Args:
        snapshot: Snapshot directory path.

    Returns:
        Path to the zip file, or ``None``.
    """
    zp = snapshot / "vox1" / "vox1_dev_wav.zip"
    return zp if zp.is_file() else None


def _read_meta_csv(snapshot: Path) -> list[tuple[str, str]]:
    """Parse the VoxCeleb1 metadata CSV.

    Returns:
        List of ``(speaker_id, name)`` tuples for dev set speakers.
    """
    meta_path = snapshot / "vox1" / "vox1_meta.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"vox1_meta.csv not found at {meta_path}")

    speakers: list[tuple[str, str]] = []
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("VoxCeleb1 ID"):
                continue
            parts = line.split("\t")
            if len(parts) >= 5 and parts[4] == "dev":
                speakers.append((parts[0], parts[1]))
    return speakers


# --------------------------------------------------------------------------- #
#  Embedding extraction
# --------------------------------------------------------------------------- #


def _extract_audio_from_zip(
    zf: zipfile.ZipFile, speaker_id: str
) -> tuple[np.ndarray, int] | None:
    """Extract the first WAV for a speaker from the combined zip.

    WAV files in the zip are structured as::

        wav/id10001/1zcIwhmdeo4/00001.wav

    Args:
        zf: Opened ZipFile.
        speaker_id: Speaker ID like ``id10001``.

    Returns:
        ``(audio_array, sample_rate)`` tuple, or ``None`` if no WAV found.
    """
    import soundfile as sf

    prefix = f"wav/{speaker_id}/"
    for name in zf.namelist():
        if name.startswith(prefix) and name.endswith(".wav"):
            try:
                with zf.open(name) as f:
                    data, sr = sf.read(f)
                return data, sr
            except Exception:
                continue
    return None


def _extract_embedding_from_wav(
    model: object, audio_data: np.ndarray, sample_rate: int
) -> np.ndarray | None:
    """Extract a 256-dim speaker embedding from in-memory audio.

    Args:
        model: Loaded wespeaker SpeakerModel.
        audio_data: Audio samples as float32 array.
        sample_rate: Sample rate in Hz.

    Returns:
        256-dim float32 embedding, or None if no speech detected.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        import soundfile as sf

        sf.write(tmp_path, audio_data, sample_rate)
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
        description="Extract cohort embeddings from downloaded VoxCeleb1 snapshot.",
    )
    parser.add_argument(
        "--voxceleb",
        type=int,
        default=300,
        help="Number of VoxCeleb speakers to include (default: 300).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="asset/cohort/cohort_embeddings.npy",
        help="Output .npy path (default: asset/cohort/cohort_embeddings.npy).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args(argv)
    output_path = Path(args.output)
    json_path = output_path.with_suffix(".json")

    # -- Find snapshot ----------------------------------------------------- #
    snapshot = _find_snapshot()
    if snapshot is None:
        logger.error(
            "ProgramComputer/voxceleb snapshot not found in HF cache.\n"
            "Run snapshot_download first."
        )
        sys.exit(1)

    logger.info("Found snapshot: %s", snapshot)

    # -- Find zip data ----------------------------------------------------- #
    full_zip = _find_full_zip(snapshot)
    zip_parts = _find_zip_parts(snapshot)

    if not full_zip and not zip_parts:
        logger.error(
            "No zip files found in snapshot.\n"
            "The download may not have completed yet."
        )
        sys.exit(1)

    # -- Parse metadata ---------------------------------------------------- #
    speakers = _read_meta_csv(snapshot)
    logger.info("Found %d speakers in dev set", len(speakers))

    # -- Dry-run ----------------------------------------------------------- #
    if args.dry_run:
        logger.info(
            "Would process %d VoxCeleb speakers → %s",
            min(args.voxceleb, len(speakers)),
            output_path,
        )
        return

    # -- Check output ------------------------------------------------------ #
    if output_path.exists() and not args.force:
        logger.error(
            "Output file already exists: %s\nUse --force to overwrite.",
            output_path,
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Load model -------------------------------------------------------- #
    logger.info("Loading VBLINKF model ...")
    model = _load_model()
    logger.info("Model loaded.")

    # -- Open zip ---------------------------------------------------------- #
    zip_handle: zipfile.ZipFile | None = None
    temp_combined: Path | None = None

    try:
        if full_zip:
            logger.info("Using full zip: %s", full_zip)
            zip_handle = zipfile.ZipFile(str(full_zip))
        elif zip_parts:
            logger.info("Combining %d zip parts ...", len(zip_parts))
            temp_combined = Path(tempfile.mktemp(suffix=".zip"))
            with open(temp_combined, "wb") as out:
                for part in zip_parts:
                    out.write(part.read_bytes())
            zip_handle = zipfile.ZipFile(str(temp_combined))

        assert zip_handle is not None

        # -- Process speakers ---------------------------------------------- #
        embeddings: list[np.ndarray] = []
        speaker_names: list[str] = []
        max_speakers = min(args.voxceleb, len(speakers))

        for i, (speaker_id, name) in enumerate(speakers):
            if len(embeddings) >= max_speakers:
                break

            audio = _extract_audio_from_zip(zip_handle, speaker_id)
            if audio is None:
                logger.debug("No WAV found for %s (%s); skipping", speaker_id, name)
                continue

            audio_data, sr = audio
            emb = _extract_embedding_from_wav(model, audio_data, sr)
            if emb is None:
                logger.debug("No speech in %s (%s); skipping", speaker_id, name)
                continue

            embeddings.append(emb)
            speaker_names.append(name)

            if len(embeddings) % 50 == 0:
                logger.info("  ... %d / %d embeddings", len(embeddings), max_speakers)

        # -- Save ---------------------------------------------------------- #
        if not embeddings:
            logger.error("No embeddings were extracted.")
            sys.exit(1)

        emb_array = np.stack(embeddings)
        np.save(str(output_path), emb_array)

        import json

        meta = {
            "embedding_dim": 256,
            "total_speakers": len(embeddings),
            "source": "ProgramComputer/voxceleb",
            "speakers": speaker_names,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            "Saved cohort: %s  (shape=%s)",
            output_path,
            emb_array.shape,
        )

    finally:
        if zip_handle is not None:
            zip_handle.close()
        if temp_combined is not None and temp_combined.exists():
            temp_combined.unlink()


if __name__ == "__main__":
    main()
