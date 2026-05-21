#!/usr/bin/env python3
"""Build cohort embeddings for AS-Norm from HuggingFace datasets.

Downloads one utterance per speaker from VoxCeleb1 and CN-Celeb,
extracts 256-dim speaker embeddings using the VBLINKF model
(vendored wespeaker, same as WespeakerDeep), and saves the cohort
as a (N, 256) float32 .npy file with companion .json metadata.

The resulting cohort is loaded by :class:`CohortCache` in the
AS-Norm pipeline (``asnorm.py``).

Usage:
    export HF_TOKEN='hf_...'
    python scripts/build_cohort.py --voxceleb 300 --cn-celeb 200
    python scripts/build_cohort.py --voxceleb 300 --cn-celeb 200 --force
    python scripts/build_cohort.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_cohort")


# --------------------------------------------------------------------------- #
#  Vendored wespeaker loading (identical to WespeakerDeep)
# --------------------------------------------------------------------------- #

def _load_model() -> object:
    """Load the VBLINKF model via the vendored ``_wespeaker/`` package.

    Uses the same ``importlib``-based module injection as
    :class:`wespeaker_deep_edge.wespeaker_deep_dege.WespeakerDeep`.

    Returns:
        A wespeaker ``SpeakerModel`` instance with ``extract_embedding()``.
    """
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
            raise ImportError(
                f"Cannot load vendored wespeaker module: {wespeaker_dir}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules["wespeaker"] = module
        spec.loader.exec_module(module)

    import wespeaker  # type: ignore[import-untyped]  # noqa: F811

    model_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "wespeaker_deep_edge"
        / "_models"
        / "vblinkf"
    )
    logger.info("Loading VBLINKF model from %s", model_dir)
    model = wespeaker.load_model(str(model_dir), dtype="float32")
    return model


# --------------------------------------------------------------------------- #
#  HuggingFace helpers
# --------------------------------------------------------------------------- #

def _get_hf_token() -> str:
    """Read ``HF_TOKEN`` from the environment.

    Returns:
        The HuggingFace token string.

    Raises:
        RuntimeError: If the environment variable is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set.\n\n"
            "To build the cohort you need a HuggingFace authentication token:\n\n"
            "  1. Create an account at https://huggingface.co/join\n"
            "  2. Generate a token at https://huggingface.co/settings/tokens\n"
            "  3. Accept the terms for VoxCeleb1:\n"
            "       https://huggingface.co/datasets/voxceleb1\n"
            "  4. (If using CN-Celeb) Accept its terms if required.\n"
            "  5. Export the token:\n\n"
            "       export HF_TOKEN='hf_your_token_here'\n\n"
            "Then re-run this script."
        )
    return token


def _check_optional_deps() -> None:
    """Verify that ``datasets`` is importable (lazy dependency)."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        logger.error(
            "The 'datasets' library is required but not installed.\n"
            "It is not a project dependency, so you need to install it:\n\n"
            "    uv add datasets huggingface_hub\n\n"
            "Or with pip:\n\n"
            "    pip install datasets huggingface_hub\n"
        )
        sys.exit(1)


def _stream_dataset(
    dataset_name: str,
    split: str,
    hf_token: str,
    max_speakers: int,
) -> list[tuple[np.ndarray, int, str]]:
    """Stream a HuggingFace dataset, collecting one utterance per speaker.

    Args:
        dataset_name: HF dataset identifier (e.g. ``"voxceleb1"``).
        split: Dataset split (e.g. ``"train"``).
        hf_token: HuggingFace authentication token.
        max_speakers: Maximum number of speakers to collect.

    Returns:
        List of ``(audio_array, sample_rate, speaker_id)`` tuples.

    Raises:
        RuntimeError: If the dataset cannot be loaded.
    """
    from datasets import load_dataset

    logger.info(
        "Loading dataset %s (split=%s, max_speakers=%d)",
        dataset_name, split, max_speakers,
    )

    try:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}': {exc}\n\n"
            "Possible causes:\n"
            f"  - You have not accepted the terms for {dataset_name}.\n"
            "  - Your HF_TOKEN is invalid or expired.\n"
            "  - The dataset name is incorrect.\n"
            "Visit https://huggingface.co/datasets and log in to accept terms."
        ) from exc

    seen_speakers: set[str] = set()
    results: list[tuple[np.ndarray, int, str]] = []
    count = 0

    for example in ds:
        speaker_id = example.get("speaker_id")
        if speaker_id is None and "label" in example:
            speaker_id = example["label"]
        if speaker_id is None:
            continue

        speaker_key = str(speaker_id)
        if speaker_key in seen_speakers:
            continue
        seen_speakers.add(speaker_key)

        audio = example.get("audio")
        if audio is None:
            continue

        audio_array: np.ndarray = audio["array"]
        sample_rate: int = audio["sampling_rate"]

        results.append((audio_array, sample_rate, speaker_key))
        count += 1
        if count >= max_speakers:
            break

        if count % 50 == 0:
            logger.info("  ... %d / %d speakers", count, max_speakers)

    logger.info("  Collected %d speakers from %s", count, dataset_name)
    return results


# --------------------------------------------------------------------------- #
#  Embedding extraction
# --------------------------------------------------------------------------- #

def _extract_embedding(
    model: object,
    audio_array: np.ndarray,
    sample_rate: int,
) -> np.ndarray | None:
    """Extract a 256-dim speaker embedding from raw audio.

    Writes the audio to a temporary WAV file, runs the model's
    ``extract_embedding()`` method, and cleans up.

    Args:
        model: Loaded wespeaker ``SpeakerModel``.
        audio_array: Audio samples (1-D float32 or int16 array).
        sample_rate: Sample rate in Hz.

    Returns:
        256-dim float32 embedding, or ``None`` if no speech was detected.
    """
    import soundfile as sf

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_path, audio_array, sample_rate)
        emb_tensor = model.extract_embedding(tmp_path)  # type: ignore[union-attr]
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build cohort embeddings for AS-Norm from HuggingFace datasets.",
    )
    parser.add_argument(
        "--voxceleb",
        type=int,
        default=300,
        help="Number of VoxCeleb speakers to include (default: 300).",
    )
    parser.add_argument(
        "--cn-celeb",
        type=int,
        default=200,
        help="Number of CN-Celeb speakers to include (default: 200).",
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
    """Build the cohort embedding database."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args(argv)

    output_path = Path(args.output)
    json_path = output_path.with_suffix(".json")

    total_expected = args.voxceleb + args.cn_celeb

    # -- Print plan ------------------------------------------------------- #
    plan_lines = [
        "Cohort build plan:",
        "  Source datasets:",
    ]
    if args.voxceleb > 0:
        plan_lines.append(f"    - VoxCeleb1:        {args.voxceleb:>4d} speakers")
    if args.cn_celeb > 0:
        plan_lines.append(f"    - CN-Celeb:         {args.cn_celeb:>4d} speakers")
    plan_lines.append("    ─" * 33)
    plan_lines.append(f"    Total:              {total_expected:>4d} speakers")
    plan_lines.append(f"  Output:  {output_path.resolve()}")
    plan_lines.append(f"  Metadata: {json_path.resolve()}")
    logger.info("\n".join(plan_lines))

    # -- Check output exists ---------------------------------------------- #
    if output_path.exists() and not args.force and not args.dry_run:
        logger.error(
            "Output file already exists: %s\n"
            "Use --force to overwrite or choose a different --output path.",
            output_path,
        )
        sys.exit(1)

    # -- Dry-run ---------------------------------------------------------- #
    if args.dry_run:
        logger.info("Dry-run mode: no work performed.")
        if not output_path.parent.exists():
            logger.info("  Would create directory: %s", output_path.parent)
        logger.info(
            "  Would download %d utterances (%d from VoxCeleb1, %d from CN-Celeb), "
            "extract 256-dim embeddings, and save.",
            total_expected, args.voxceleb, args.cn_celeb,
        )
        return

    # -- Create output directory ------------------------------------------ #
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Load model ------------------------------------------------------- #
    logger.info("Loading VBLINKF model ...")
    model = _load_model()
    logger.info("Model loaded successfully.")

    # -- Check optional deps ---------------------------------------------- #
    _check_optional_deps()

    # -- Get HF token ----------------------------------------------------- #
    try:
        hf_token = _get_hf_token()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # -- Stream datasets and extract embeddings --------------------------- #
    all_embeddings: list[np.ndarray] = []
    metadata: dict = {
        "embedding_dim": 256,
        "total_speakers": 0,
        "sources": [],
        "speakers": {},
    }

    counts = [
        ("voxceleb1", "train", args.voxceleb, "VoxCeleb1"),
        ("kpyu/cn-celeb", "train", args.cn_celeb, "CN-Celeb"),
    ]

    for dataset_name, split, max_speakers, label in counts:
        if max_speakers <= 0:
            continue

        try:
            utterances = _stream_dataset(
                dataset_name, split, hf_token, max_speakers,
            )
        except RuntimeError as exc:
            logger.warning("Skipping %s: %s", label, exc)
            continue

        source_speakers: list[str] = []
        for audio_array, sample_rate, speaker_key in utterances:
            emb = _extract_embedding(model, audio_array, sample_rate)
            if emb is None:
                logger.debug("No speech in utterance for speaker %s; skipping.", speaker_key)
                continue
            all_embeddings.append(emb)
            source_speakers.append(speaker_key)

        if source_speakers:
            metadata["sources"].append({
                "dataset": dataset_name,
                "split": split,
                "label": label,
                "count": len(source_speakers),
                "speakers": source_speakers,
            })

        logger.info(
            "%s: extracted %d / %d embeddings",
            label, len(source_speakers), max_speakers,
        )

    # -- Validate and save ------------------------------------------------ #
    if not all_embeddings:
        logger.error("No embeddings were extracted. Cohort file was not created.")
        sys.exit(1)

    embeddings_array = np.stack(all_embeddings)  # (N, 256)
    np.save(str(output_path), embeddings_array)

    metadata["total_speakers"] = len(all_embeddings)
    metadata["embedding_dim"] = 256
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Cohort saved: %s  (shape=%s, dtype=%s)",
        output_path,
        embeddings_array.shape,
        embeddings_array.dtype,
    )


if __name__ == "__main__":
    main()
