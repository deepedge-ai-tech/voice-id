#!/usr/bin/env python3
"""Classify audio files in asset/orin/ using all built-in voiceprints.

For each audio file, extracts embedding once, compares against all 8 voiceprints,
and copies to asset_filter/<person>/ based on best match.

Usage:
    uv run python scripts/classify_orin.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from wespeaker_deep_edge._voiceprints import (
    _PEOPLE,
    get_voiceprint_path,
    get_voiceprint_name,
)
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep

SRC = Path("asset/orin/registration_segments")
DST = Path("asset_filter")
THRESHOLD = 0.30  # minimum confidence to assign a label


def main() -> None:
    if not SRC.is_dir():
        print(f"错误: 目录不存在 {SRC}")
        sys.exit(1)

    # -- init model once --
    recognizer = WespeakerDeep()

    # -- load all reference voiceprints --
    refs: list[tuple[int, torch.Tensor]] = []
    for i in range(len(_PEOPLE)):
        emb = recognizer.load(get_voiceprint_path(i))
        refs.append((i, torch.from_numpy(emb.astype(np.float32))))
        print(f"  [{i}] {get_voiceprint_name(i)}")

    audio_files = sorted(SRC.glob("*.wav"))
    print(f"\n共 {len(audio_files)} 个音频文件\n")

    stats: dict[str, int] = {name: 0 for name in _PEOPLE}
    stats["_unknown"] = 0

    for af in audio_files:
        # extract embedding once
        test_emb = recognizer._model.extract_embedding(str(af))
        if test_emb is None:
            print(f"  ⚠ 无有效语音: {af.name}")
            _copy_to(af, DST / "_unknown")
            stats["_unknown"] += 1
            continue

        # find best match
        scores: list[tuple[int, float]] = []
        for idx, ref_emb in refs:
            score = recognizer._model.cosine_similarity(test_emb, ref_emb)
            scores.append((idx, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_score = scores[0]

        if best_score >= THRESHOLD:
            name = get_voiceprint_name(best_idx)
            _copy_to(af, DST / name)
            stats[name] += 1
            flag = "  "
        else:
            _copy_to(af, DST / "_unknown")
            stats["_unknown"] += 1
            flag = "?"

        print(
            f"  {flag} {af.name}"
            f"  → {get_voiceprint_name(best_idx):>12}  {best_score:.4f}"
            f"  (top-3: {', '.join(f'{get_voiceprint_name(s[0])}:{s[1]:.3f}' for s in scores[:3])})"
        )

    # -- summary --
    print("\n分类结果:")
    for name in _PEOPLE:
        print(f"  {name:<16}: {stats[name]}")
    print(f"  {'_unknown':<16}: {stats['_unknown']}")


def _copy_to(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst_dir / src.name))


if __name__ == "__main__":
    main()
