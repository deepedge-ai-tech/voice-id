"""批量注册声纹 → 生成内置 .pkl 文件到 _voiceprints/。

用法:
    uv run python scripts/batch_enroll_voiceprints.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torchaudio
from wespeaker_deep_edge._utils import _load_audio
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep

ASSET = Path(__file__).resolve().parent.parent / "asset"
DST = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "wespeaker_deep_edge"
    / "_voiceprints"
)

PEOPLE = {
    "john": "John",
    "frank": "Frank",
    "michael": "Michael",
    "qingqing": "Qingqing",
    "xixi": "Xixi",
    "zhong": "Zhong",
}


def enroll_person(person_dir: str, name_en: str) -> None:
    seg_dir = ASSET / person_dir / "registration_segments"
    if not seg_dir.is_dir():
        print(f"  [SKIP] {seg_dir} not found")
        return

    wavs = sorted(seg_dir.glob("*.wav"))
    if not wavs:
        print(f"  [SKIP] no wav files in {seg_dir}")
        return

    print(f"  Concatenating {len(wavs)} segments ...")
    parts = [_load_audio(str(w)) for w in wavs]
    combined = torch.cat(parts)

    tmp = DST / f"_{name_en.lower()}_combined.wav"
    torchaudio.save(str(tmp), combined.unsqueeze(0), 16000)

    deep = WespeakerDeep()
    out_path = DST / f"voice_{name_en.lower()}.pkl"
    result = deep.enroll(str(tmp), str(out_path))
    tmp.unlink()
    print(f"  Result: {result}")
    print(f"  Done → {out_path.name}")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for person_dir, name_en in PEOPLE.items():
        print(f"\n=== {name_en} ({person_dir}) ===")
        enroll_person(person_dir, name_en)
    print("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()
