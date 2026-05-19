#!/usr/bin/env python3
"""Official WeSpeaker N×N cross-test with VAD, English model, and heatmap.

Usage:
    cd Voice-ID
    uv run python scripts/official_cross_test.py

Output:
    - Console: per-person stats and same/different gap
    - Image: scripts/output/official_cross_test_heatmap.png
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wespeaker

# ── Configuration ────────────────────────────────────────────────────────────

PEOPLE = ["john", "frank", "michael", "qingqing", "xixi", "zhong"]
ASSET = Path("asset")
ASSET_COMBINE = Path("asset_combine")
OUTPUT_DIR = Path("scripts/output")

# Enrollment uses asset_combine/{Name}.wav (single full-duration file per person)
# Test files use asset/{person}/test_segments/*.wav
ENROLL_SOURCES: dict[str, Path] = {
    p: ASSET_COMBINE / f"{p.capitalize()}.wav" for p in PEOPLE
}

TEST_DIRS: dict[str, str | None] = {
    "john": "test_segments",
    "frank": "test_segments",
    "michael": None,
    "qingqing": "test_segments",
    "xixi": "test_segments",
    "zhong": "test_segments",
}

# Modification notes — edit this before each run to document changes
ENROLL_NOTES: str = "Enrollment: asset_combine/*.wav (single file per person)"

NUM_MEL_BINS = 80
FRAME_LENGTH = 25
FRAME_SHIFT = 10
CMN = True


# ── Helpers ──────────────────────────────────────────────────────────────────


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Cosine similarity between two 1-D embedding vectors."""
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))


PATCHED = False


def load_model(
    model_name: str = "vblinkf",
    vad: bool = True,
    num_mel_bins: int = NUM_MEL_BINS,
    frame_length: int = FRAME_LENGTH,
    frame_shift: int = FRAME_SHIFT,
    cmn: bool = CMN,
):
    """Load WeSpeaker model and patch fbank parameters."""
    global PATCHED
    model = wespeaker.load_model(model_name)

    import torch
    from torchaudio.compliance import kaldi

    if not PATCHED:

        def _patched_compute(wavform, sample_rate=16000, cmn=cmn):
            feat = kaldi.fbank(
                wavform,
                num_mel_bins=num_mel_bins,
                frame_length=frame_length,
                frame_shift=frame_shift,
                sample_frequency=sample_rate,
                window_type=model.window_type,
            )
            if cmn:
                feat = feat - torch.mean(feat, dim=0)
            return feat.unsqueeze(0)

        model.compute_features = _patched_compute
        PATCHED = True

    if vad:
        model.set_vad(True)
    return model


# ── Speaker ──────────────────────────────────────────────────────────────────


class Speaker:
    """One enrolled speaker.

    Usage::

        model = load_model()
        alice = Speaker.enroll("alice", "alice.wav", model)
        bob = Speaker.enroll("bob", "bob.wav", model)

        score = alice.recognize("test.wav", model)       # single recognition
        sim = alice.similarity(bob)                       # speaker vs speaker
    """

    def __init__(self, name: str, embedding: np.ndarray):
        self.name = name
        self.embedding = embedding

    @classmethod
    def enroll(cls, name: str, audio_path: Path | str, model) -> "Speaker":
        """Create a Speaker by extracting embedding from an audio file."""
        emb = model.extract_embedding(str(audio_path))
        if emb is None:
            raise ValueError(f"VAD filtered all audio in {audio_path}")
        return cls(name, emb)

    def similarity(self, other: "Speaker") -> float:
        """Cosine similarity between this and another Speaker."""
        return cosine_similarity(self.embedding, other.embedding)

    def recognize(self, audio_path: Path | str, model) -> float:
        """Recognize audio against this speaker. Returns similarity score."""
        emb = model.extract_embedding(str(audio_path))
        if emb is None:
            return 0.0
        return cosine_similarity(self.embedding, emb)


# ── Cross-test ───────────────────────────────────────────────────────────────


@dataclass
class CrossTestResult:
    """Results from run_cross_test()."""

    speakers: list[Speaker]
    test_files_map: dict[str, list[Path]]
    test_embs: dict[str, list[np.ndarray]]
    per_file_results: dict[str, list[tuple[str, list[float]]]]
    sim_matrix: np.ndarray
    same_mean: float
    diff_mean: float
    gap: float


def run_cross_test(
    speakers: list[Speaker],
    test_files_map: dict[str, list[Path]],
    model,
) -> CrossTestResult:
    """Run N×N cross-test: each test file against every enrolled speaker.

    Args:
        speakers: list of enrolled Speaker objects.
        test_files_map: {person_name: [audio_paths]}.
        model: loaded WeSpeaker model.

    Returns:
        CrossTestResult with similarity matrix and per-file scores.
    """
    name_to_speaker = {s.name: s for s in speakers}
    active_people = sorted(
        p for p in test_files_map if p in name_to_speaker and test_files_map[p]
    )

    # Extract test embeddings
    test_embs: dict[str, list[np.ndarray]] = {}
    for person, files in test_files_map.items():
        if person not in name_to_speaker:
            continue
        embs = [model.extract_embedding(str(f)) for f in files]
        valid = [e for e in embs if e is not None]
        if len(valid) < len(embs):
            print(
                f"  {person}: {len(embs) - len(valid)} files returned None"
                f" (VAD filtered), keeping {len(valid)}"
            )
        test_embs[person] = valid
        print(f"  Extracted {len(valid)} embeddings for {person}")

    active_people = sorted(p for p in active_people if test_embs.get(p))
    N = len(active_people)

    # Per-file similarity
    per_file_results: dict[str, list[tuple[str, list[float]]]] = {}
    for test_person in active_people:
        entries = []
        for test_file, test_emb in zip(
            test_files_map[test_person], test_embs[test_person]
        ):
            scores = [
                cosine_similarity(test_emb, name_to_speaker[enroll_p].embedding)
                for enroll_p in active_people
            ]
            entries.append((test_file.name, scores))
        per_file_results[test_person] = entries

    # Aggregate matrix
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    for i, tp in enumerate(active_people):
        for j in range(N):
            vals = [e[1][j] for e in per_file_results[tp]]
            sim_matrix[i][j] = np.mean(vals)

    same_scores = [sim_matrix[i][i] for i in range(N)]
    diff_scores = [
        sim_matrix[i][j] for i in range(N) for j in range(N) if i != j
    ]
    same_mean = float(np.mean(same_scores)) * 100
    diff_mean = float(np.mean(diff_scores)) * 100

    return CrossTestResult(
        speakers=speakers,
        test_files_map=test_files_map,
        test_embs=test_embs,
        per_file_results=per_file_results,
        sim_matrix=sim_matrix,
        same_mean=same_mean,
        diff_mean=diff_mean,
        gap=same_mean - diff_mean,
    )


# ── Result rendering ────────────────────────────────────────────────────────


def print_results(result: CrossTestResult) -> None:
    """Print detailed per-file results and summary to console."""
    active_people = sorted(result.per_file_results.keys())
    N = len(active_people)
    enroll_labels = [p.capitalize() for p in active_people]

    print("\n" + "=" * 90)
    print("Detailed Per-File Results")
    print("=" * 90)

    header = (
        f"{'Test Person':14s} {'Test File':30s}"
        + "".join(f"{e:>9s}" for e in enroll_labels)
    )
    print(header)
    print("-" * 90)

    for test_person in active_people:
        for fname, scores in result.per_file_results[test_person]:
            scores_pct = "".join(f"{s * 100:>8.1f}%" for s in scores)
            print(f"{test_person:14s} {fname:30s} {scores_pct}")
        print()

    print("=" * 90)
    print("Summary Statistics")
    print("=" * 90)
    print(f"\n  Same-person mean:     {result.same_mean:.1f}%")
    print(f"  Different-person mean: {result.diff_mean:.1f}%")
    print(f"  Gap:                   {result.gap:.1f}%")

    print(f"\n  Per-person (same-person only):")
    for i, p in enumerate(active_people):
        n = len(result.per_file_results[p])
        scores_self = [e[1][i] * 100 for e in result.per_file_results[p]]
        scores_other = [
            e[1][j] * 100
            for e in result.per_file_results[p]
            for j in range(N)
            if j != i
        ]
        other_mean = np.mean(scores_other) if scores_other else 0.0
        print(
            f"    {p:12s}: self={np.mean(scores_self):5.1f}%"
            f"  other={other_mean:5.1f}%"
            f"  gap={np.mean(scores_self) - other_mean:5.1f}%"
            f"  n={n}"
        )


def save_heatmaps(
    result: CrossTestResult, ts: str, output_dir: Path = OUTPUT_DIR
) -> tuple[Path, Path]:
    """Save per-file and aggregate similarity heatmaps to disk.

    Returns:
        (per_file_path, aggregate_path)
    """
    active_people = sorted(result.per_file_results.keys())
    N = len(active_people)
    enroll_labels = [p.capitalize() for p in active_people]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-file heatmap
    row_labels: list[str] = []
    per_file_matrix: list[list[float]] = []
    for test_person in active_people:
        for fname, scores in result.per_file_results[test_person]:
            row_labels.append(f"{test_person}/{fname}")
            per_file_matrix.append(scores)

    per_file_arr = np.array(per_file_matrix, dtype=np.float32) * 100
    n_rows = len(row_labels)

    fig_height = max(6, n_rows * 0.35)
    plt.figure(figsize=(12, fig_height))
    sns.heatmap(
        per_file_arr,
        xticklabels=enroll_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Cosine Similarity (%)"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(
        "WeSpeaker Per-File Cross-Test (VoxBlink2 SAM-ResNet34 + VAD)",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Enrollment Speaker", fontsize=12)
    plt.ylabel("Test File", fontsize=12)
    plt.tight_layout()

    per_file_path = output_dir / f"cross_test_per_file_{ts}.png"
    plt.savefig(per_file_path, dpi=150)
    print(f"\n  Per-file heatmap saved to: {per_file_path}")
    plt.close()

    # Aggregate heatmap
    plt.figure(figsize=(10, 8))
    annot_agg = [
        [f"{result.sim_matrix[i][j] * 100:.1f}%" for j in range(N)]
        for i in range(N)
    ]
    sns.heatmap(
        result.sim_matrix * 100,
        xticklabels=enroll_labels,
        yticklabels=enroll_labels,
        annot=annot_agg,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Cosine Similarity (%)"},
        linewidths=1,
        linecolor="white",
    )
    plt.title(
        "WeSpeaker Cross-Test Aggregate (English Model + VAD)",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Enrollment Speaker", fontsize=12)
    plt.ylabel("Test Speaker (avg)", fontsize=12)
    plt.tight_layout()

    agg_path = output_dir / f"cross_test_aggregate_{ts}.png"
    plt.savefig(agg_path, dpi=150)
    print(f"  Aggregate heatmap saved to: {agg_path}")
    plt.close()

    return per_file_path, agg_path


# ── Report ────────────────────────────────────────────────────────────────────


def save_report(
    result: CrossTestResult,
    ts: str,
    per_file_img: Path,
    agg_img: Path,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save a markdown report with modification notes and results table."""
    active_people = sorted(result.per_file_results.keys())
    N = len(active_people)
    enroll_labels = [p.capitalize() for p in active_people]

    lines = [
        f"# Cross-Test Report — {ts}",
        "",
        f"**Enrollment source**: `asset_combine/` (single file per person)",
        f"**Test source**: `asset/{{person}}/test_segments/`",
        f"**Model**: vblinkf (VAD+ CMN={CMN} mel={NUM_MEL_BINS})",
        "",
        "## Modifications",
        "",
        ENROLL_NOTES,
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Same-person mean | {result.same_mean:.1f}% |",
        f"| Different-person mean | {result.diff_mean:.1f}% |",
        f"| Gap | {result.gap:.1f}% |",
        "",
        "## Per-Person",
        "",
        "| Person | Self | Other | Gap | Tests |",
        "|--------|------|-------|-----|-------|",
    ]

    for i, p in enumerate(active_people):
        n = len(result.per_file_results[p])
        scores_self = [e[1][i] * 100 for e in result.per_file_results[p]]
        scores_other = [
            e[1][j] * 100
            for e in result.per_file_results[p]
            for j in range(N)
            if j != i
        ]
        other_mean = np.mean(scores_other) if scores_other else 0.0
        self_mean = np.mean(scores_self)
        gap = self_mean - other_mean
        lines.append(
            f"| {p.capitalize()} | {self_mean:.1f}% | {other_mean:.1f}% | {gap:.1f}% | {n} |"
        )

    lines.extend(
        [
            "",
            "## Per-File Details",
            "",
            f"Test Person | Test File | " + " | ".join(enroll_labels) + " |",
            "|---|" + "---|" * (N + 1),
        ]
    )

    for test_person in active_people:
        for fname, scores in result.per_file_results[test_person]:
            scores_str = " | ".join(f"{s * 100:.1f}%" for s in scores)
            lines.append(f"{test_person} | {fname} | {scores_str}")

    lines.extend(
        [
            "",
            "## Heatmaps",
            "",
            f"![Per-file heatmap]({per_file_img.name})",
            "",
            f"![Aggregate heatmap]({agg_img.name})",
            "",
        ]
    )

    report_path = output_dir / f"cross_test_report_{ts}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Report saved to: {report_path}")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(
        f"WeSpeaker Cross-Test: vblinkf VAD+ cmn={CMN}"
        f" mel={NUM_MEL_BINS} fl={FRAME_LENGTH} fs={FRAME_SHIFT}"
    )
    print(f"  Enroll from: asset_combine/")
    print("=" * 60)

    # 1. Load model
    print("\n[1/4] Loading English model with VAD enabled...")
    t0 = time.time()
    model = load_model()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # 2. Collect files & enroll
    print("\n[2/4] Collecting enrollment and test files...")
    speakers: list[Speaker] = []
    test_files_map: dict[str, list[Path]] = {}
    for person in PEOPLE:
        reg = ENROLL_SOURCES[person]
        if not reg.exists():
            print(f"  WARNING: {reg} not found, skipping {person}")
            continue

        try:
            speaker = Speaker.enroll(person, reg, model)
            speakers.append(speaker)
            print(f"  {person}: enroll={reg.name}, done")
        except ValueError as e:
            print(f"  WARNING: {e}, skipping {person}")
            continue

        base = ASSET / person
        test_dir = TEST_DIRS[person]
        tests = (
            sorted(base.glob(f"{test_dir}/*.wav"))
            if test_dir
            else [base / "测试.wav"]
        )
        if tests:
            test_files_map[person] = tests
            print(f"  {person}: tests={len(tests)} files")

    print(f"\n  Active people: {[s.name for s in speakers]}")

    # 3. Cross-test
    print(f"\n[3/4] Running cross-test...")
    t0 = time.time()
    result = run_cross_test(speakers, test_files_map, model)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n  Effective people for matrix: {sorted(result.per_file_results.keys())}")

    # 4. Print results
    print_results(result)

    # 5. Save heatmaps
    per_file_img, agg_img = save_heatmaps(result, ts)

    # 6. Save report
    save_report(result, ts, per_file_img, agg_img)

    # 7. Summary line
    print("\n" + "-" * 60)
    print(
        f"SUMMARY: Same={result.same_mean:.1f}%"
        f", Diff={result.diff_mean:.1f}%"
        f", Gap={result.gap:.1f}%"
    )
    print("-" * 60)


if __name__ == "__main__":
    main()
