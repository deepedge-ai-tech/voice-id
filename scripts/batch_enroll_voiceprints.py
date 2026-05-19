"""批量注册声纹 → 生成内置 .pkl 文件到 _voiceprints/。"""

import sys
from pathlib import Path

# 确保能找到 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torchaudio
import torch
from wespeaker_deep_edge.client import WespeakerClient, _load_audio


ASSET = Path(__file__).resolve().parent.parent / "asset"
DST = Path(__file__).resolve().parent.parent / "src" / "wespeaker_deep_edge" / "_voiceprints"

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
    # 加载所有 segment 并拼接
    parts = []
    for w in wavs:
        parts.append(_load_audio(str(w)))
    combined = torch.cat(parts)  # shape: (N,)

    # 保存临时文件
    tmp = DST / f"_{name_en.lower()}_combined.wav"
    torchaudio.save(str(tmp), combined.unsqueeze(0), 16000)

    # 注册
    client = WespeakerClient(
        enable_augmentation=False,
        enable_multi_scale_enrollment=True,
    )
    out_path = DST / f"voice_{name_en.lower()}.pkl"
    result = client.mp3_to_pk(str(tmp), str(out_path))
    print(f"  Result: {result}")
    tmp.unlink()  # 删除临时文件
    print(f"  Done → {out_path.name}")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for person_dir, name_en in PEOPLE.items():
        print(f"\n=== {name_en} ({person_dir}) ===")
        enroll_person(person_dir, name_en)
    print("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()
