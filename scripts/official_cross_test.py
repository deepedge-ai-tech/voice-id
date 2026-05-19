#!/usr/bin/env python3
"""Official WeSpeaker N×N cross-test with VAD, English model, async execution, and heatmap.

Usage:
    cd Voice-ID
    uv run python scripts/official_cross_test.py

Output:
    - Console: per-person stats and same/different gap
    - Image: scripts/output/official_cross_test_heatmap.png
"""

import asyncio
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless use
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wespeaker

# ── Configuration ────────────────────────────────────────────────────────────

PEOPLE = ["john", "frank", "michael", "qingqing", "xixi", "zhong"]
ASSET = Path("asset")
OUTPUT_DIR = Path("scripts/output")

# Registration file per person (frank uses 2-digit, others 3-digit)
REG_FILES = {
    "john": "registration_segments/segment_002.wav",
    "frank": "registration_segments/segment_02.wav",
    "michael": "registration_segments/segment_002.wav",
    "qingqing": "registration_segments/segment_002.wav",
    "xixi": "registration_segments/segment_002.wav",
    "zhong": "registration_segments/segment_002.wav",
}

# Test segment sources (michael has no test_segments/)
TEST_DIRS = {
    "john": "test_segments",
    "frank": "test_segments",
    "michael": None,  # use 测试.wav directly
    "qingqing": "test_segments",
    "xixi": "test_segments",
    "zhong": "test_segments",
}

MAX_CONCURRENT = 1  # single-thread to avoid OOM

# ── Feature extraction parameters ─────────────────────────────────────────────
# Changing num_mel_bins/frame_length/frame_shift creates train/test mismatch
# and will likely hurt accuracy. cmn is safe to toggle.
NUM_MEL_BINS = 80       # FBank feature dimension
FRAME_LENGTH = 25       # Frame length (ms)
FRAME_SHIFT = 10        # Frame shift (ms)
CMN = True              # Cepstral mean normalization


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_test_files(person: str) -> list[Path]:
    """Return sorted list of test .wav files for a person."""
    base = ASSET / person
    test_dir = TEST_DIRS[person]
    if test_dir:
        return sorted(base.glob(f"{test_dir}/*.wav"))
    else:
        return [base / "测试.wav"]


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Cosine similarity between two 1-D embedding vectors."""
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))


# ── Async embedding extraction ───────────────────────────────────────────────

async def extract_embedding_async(model, audio_path: Path, sem: asyncio.Semaphore) -> np.ndarray:
    """Run model.extract_embedding in a thread, bounded by semaphore."""
    async with sem:
        return await asyncio.to_thread(model.extract_embedding, str(audio_path))


async def extract_all_embeddings(
    model, file_map: dict[str, list[Path]]
) -> dict[str, list[np.ndarray]]:
    """Extract embeddings for all files concurrently."""
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    result: dict[str, list[np.ndarray]] = {}

    for person, files in file_map.items():
        tasks = [extract_embedding_async(model, f, sem) for f in files]
        embs = await asyncio.gather(*tasks)
        # Filter out None returns (VAD removed all audio)
        valid = [e for e in embs if e is not None]
        if len(valid) < len(embs):
            print(f"  {person}: {len(embs) - len(valid)} files returned None (VAD filtered), keeping {len(valid)}")
        result[person] = valid
        print(f"  Extracted {len(result[person])} embeddings for {person}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"WeSpeaker Cross-Test: vblinkf VAD+ cmn={CMN} mel={NUM_MEL_BINS} fl={FRAME_LENGTH} fs={FRAME_SHIFT}")
    print("=" * 60)

    # 1. Load model
    print("\n[1/4] Loading English model with VAD enabled...")
    t0 = time.time()
    model = wespeaker.load_model("vblinkf")

    # Monkey-patch compute_features to use custom fbank parameters
    import torch
    from torchaudio.compliance import kaldi
    original_compute = model.compute_features
    def patched_compute(wavform, sample_rate=16000, cmn=CMN):
        feat = kaldi.fbank(
            wavform,
            num_mel_bins=NUM_MEL_BINS,
            frame_length=FRAME_LENGTH,
            frame_shift=FRAME_SHIFT,
            sample_frequency=sample_rate,
            window_type=model.window_type,
        )
        if cmn:
            feat = feat - torch.mean(feat, dim=0)
        return feat.unsqueeze(0)
    model.compute_features = patched_compute

    model.set_vad(True)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # 2. Collect files
    print("\n[2/4] Collecting enrollment and test files...")
    enroll_files: dict[str, Path] = {}
    test_files_map: dict[str, list[Path]] = {}
    for person in PEOPLE:
        reg = ASSET / person / REG_FILES[person]
        if not reg.exists():
            print(f"  WARNING: {reg} not found, skipping {person}")
            continue
        enroll_files[person] = reg
        tests = get_test_files(person)
        if not tests:
            print(f"  WARNING: no test files for {person}, skipping")
            continue
        test_files_map[person] = tests
        print(f"  {person}: enroll={reg.name}, tests={len(tests)} files")

    active_people = list(enroll_files.keys())
    print(f"\n  Active people: {active_people}")

    # 3. Extract embeddings
    print(f"\n[3/4] Extracting embeddings (max {MAX_CONCURRENT} concurrent)...")
    t0 = time.time()

    async def run_extraction():
        # Enrollment embeddings
        enroll_embs: dict[str, np.ndarray] = {}
        sem_enroll = asyncio.Semaphore(MAX_CONCURRENT)
        enroll_tasks = {
            p: extract_embedding_async(model, f, sem_enroll)
            for p, f in enroll_files.items()
        }
        for p, task in enroll_tasks.items():
            emb = await task
            if emb is None:
                print(f"  WARNING: enroll {p} returned None (VAD filtered all audio), skipping")
                continue
            enroll_embs[p] = emb
            print(f"  Enroll {p}: done")

        # Test embeddings
        test_embs = await extract_all_embeddings(model, test_files_map)
        return enroll_embs, test_embs

    enroll_embs, test_embs = asyncio.run(run_extraction())
    print(f"  All embeddings extracted in {time.time() - t0:.1f}s")

    # Filter to people who have both valid enrollment and test embeddings
    active_people = [p for p in active_people
                     if p in enroll_embs and p in test_embs and test_embs[p]]
    N = len(active_people)
    print(f"\n  Effective people for matrix: {active_people} (N={N})")

    # 4. Compute per-file similarity
    print(f"\n[4/4] Computing similarities...")

    # Store per-file results: for each test file, scores against all enrollments
    # per_file_results[test_person] = [(filename, [scores]), ...]
    per_file_results: dict[str, list[tuple[str, list[float]]]] = {}

    for test_person in active_people:
        entries = []
        for test_file, test_emb in zip(test_files_map[test_person], test_embs[test_person]):
            scores = []
            for enroll_person in active_people:
                score = cosine_similarity(test_emb, enroll_embs[enroll_person])
                scores.append(score)
            entries.append((test_file.name, scores))
        per_file_results[test_person] = entries

    # Aggregate matrix (mean per person)
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    std_matrix = np.zeros((N, N), dtype=np.float32)
    for i, tp in enumerate(active_people):
        for j, ep in enumerate(active_people):
            vals = [e[1][j] for e in per_file_results[tp]]
            sim_matrix[i][j] = np.mean(vals)
            std_matrix[i][j] = np.std(vals)

    # 5. Print detailed per-file results
    print("\n" + "=" * 90)
    print("Detailed Per-File Results")
    print("=" * 90)

    enroll_labels = [p.capitalize() for p in active_people]
    header = f"{'Test Person':14s} {'Test File':30s}" + "".join(f"{e:>9s}" for e in enroll_labels)
    print(header)
    print("-" * 90)

    for test_person in active_people:
        for fname, scores in per_file_results[test_person]:
            scores_pct = "".join(f"{s * 100:>8.1f}%" for s in scores)
            print(f"{test_person:14s} {fname:30s} {scores_pct}")
        print()  # blank line between groups

    # 6. Stats summary
    print("=" * 90)
    print("Summary Statistics")
    print("=" * 90)

    same_scores = [sim_matrix[i][i] for i in range(N)]
    diff_scores = [sim_matrix[i][j] for i in range(N) for j in range(N) if i != j]
    same_mean = np.mean(same_scores) * 100
    diff_mean = np.mean(diff_scores) * 100
    gap = same_mean - diff_mean

    print(f"\n  Same-person mean:     {same_mean:.1f}%")
    print(f"  Different-person mean: {diff_mean:.1f}%")
    print(f"  Gap:                   {gap:.1f}%")

    print(f"\n  Per-person (same-person only):")
    for i, p in enumerate(active_people):
        n = len(per_file_results[p])
        scores_self = [e[1][i] * 100 for e in per_file_results[p]]
        scores_other = [e[1][j] * 100 for e in per_file_results[p] for j in range(N) if j != i]
        print(f"    {p:12s}: self={np.mean(scores_self):5.1f}%  other={np.mean(scores_other):5.1f}%  gap={np.mean(scores_self) - np.mean(scores_other):5.1f}%  n={n}")

    # 7. Per-file heatmap
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the per-file matrix: rows = all test files, cols = enrollments
    row_labels: list[str] = []
    per_file_matrix: list[list[float]] = []
    for test_person in active_people:
        for fname, scores in per_file_results[test_person]:
            row_labels.append(f"{test_person}/{fname}")
            per_file_matrix.append(scores)

    per_file_arr = np.array(per_file_matrix, dtype=np.float32) * 100
    n_rows = len(row_labels)

    # Row colors: alternating by person group
    row_colors = []
    colors = ["#e8f4f8", "#f0f0f0"]
    for i, tp in enumerate(active_people):
        count = len(per_file_results[tp])
        row_colors.extend([colors[i % 2]] * count)

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
    plt.title("WeSpeaker Per-File Cross-Test (VoxBlink2 SAM-ResNet34 + VAD)", fontsize=14, pad=20)
    plt.xlabel("Enrollment Speaker", fontsize=12)
    plt.ylabel("Test File", fontsize=12)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "official_cross_test_per_file_heatmap.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Per-file heatmap saved to: {out_path}")
    plt.close()

    # Also save aggregate heatmap
    plt.figure(figsize=(10, 8))
    annot_agg = [[f"{sim_matrix[i][j]*100:.1f}%" for j in range(N)] for i in range(N)]
    sns.heatmap(
        sim_matrix * 100,
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
    plt.title("WeSpeaker Cross-Test Aggregate (English Model + VAD)", fontsize=14, pad=20)
    plt.xlabel("Enrollment Speaker", fontsize=12)
    plt.ylabel("Test Speaker (avg)", fontsize=12)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "official_cross_test_heatmap.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Aggregate heatmap saved to: {out_path}")
    plt.close()

    # 8. Summary line
    print("\n" + "-" * 60)
    print(f"SUMMARY: Same={same_mean:.1f}%, Diff={diff_mean:.1f}%, Gap={gap:.1f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()
