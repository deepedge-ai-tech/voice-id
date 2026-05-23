"""CLI 入口点 — 使用 WespeakerDeep（官方 wespeaker 模型）。

支持 `python -m wespeaker_deep_edge` 调用。
"""

import argparse
import logging
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

from ._voiceprints import get_voiceprint_path, get_voiceprint_name
from .wespeaker_deep_dege import DeepConfig, WespeakerDeep

logger = logging.getLogger(__name__)

_PKG_VERSION = _pkg_version("wespeaker-deep-edge")


def _voiceprint_index_type(val: str) -> int:
    idx = int(val)
    max_idx = len(
        ["john", "frank", "michael", "qingqing", "xixi", "zhong", "angle"]
    ) - 1
    if idx < 0 or idx > max_idx:
        raise argparse.ArgumentTypeError(f"索引范围 0-{max_idx}")
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="WeSpeaker 声纹注册与识别（官方 wespeaker 模型）")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_PKG_VERSION}",
    )
    sub = parser.add_subparsers(dest="cmd")

    # ---- enroll ----
    p_reg = sub.add_parser("enroll", help="注册声纹")
    p_reg.add_argument("audio", help="音频文件路径")
    p_reg.add_argument("output", nargs="?", default="voice.pkl", help="输出 .pkl 路径")

    # ---- recognize ----
    p_rec = sub.add_parser("recognize", help="识别声纹")
    p_rec.add_argument("audio", help="音频文件路径")
    p_rec.add_argument(
        "voiceprint", nargs="?", default=None,
        help="声纹 .pkl 路径（省略则使用内置声纹）",
    )
    p_rec.add_argument(
        "--package-pk-index", type=_voiceprint_index_type, default=None,
        help="内置声纹索引: 0=John 1=Frank 2=Michael 3=Qingqing 4=Xixi 5=Zhong 6=Angle 7=Albert",
    )
    p_rec.add_argument("--threshold", type=float, default=0.70, help="相似度阈值")

    # ---- list-voiceprints ----
    p_list = sub.add_parser("list-voiceprints", help="列出所有内置声纹")

    args = parser.parse_args()

    if args.cmd == "enroll":
        deep = WespeakerDeep()
        r = deep.enroll(args.audio, args.output)
        print(r)
    elif args.cmd == "recognize":
        deep = WespeakerDeep(config=DeepConfig(sim_threshold=args.threshold))
        if args.package_pk_index is not None:
            deep.deep_config.package_pk_index = args.package_pk_index
        r = deep.recognize(args.audio, args.voiceprint)
        print(
            f"{'✅ 识别成功' if r['is_recognized'] else '❌ 未识别'}  "
            f"confidence={r['confidence']:.4f}  "
            f"threshold={r['threshold']}"
        )
    elif args.cmd == "list-voiceprints":
        names = ["john", "frank", "michael", "qingqing", "xixi", "zhong", "angle", "albert"]
        print("内置声纹列表:")
        for i, name in enumerate(names):
            print(f"  {i}: {name}  →  {get_voiceprint_path(i)}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
