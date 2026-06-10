"""CLI entry point — uses WespeakerDeep HTTP client.

Supports ``python -m wespeaker_deep_edge``.
"""

import argparse
import logging
import sys
from importlib.metadata import version as _pkg_version

from .client import WespeakerDeep

logger = logging.getLogger(__name__)

_PKG_VERSION = _pkg_version("wespeaker-deep-edge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeSpeaker 声纹注册与识别（voice-id HTTP API）"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_PKG_VERSION}",
    )
    parser.add_argument(
        "--url", default=None,
        help="voice-id API base URL (default: $VOICE_ID_URL or http://127.0.0.1:8005)",
    )
    parser.add_argument(
        "--key", default=None,
        help="API key (default: $VOICE_ID_KEY)",
    )

    sub = parser.add_subparsers(dest="cmd")

    # ---- enroll ----
    p_reg = sub.add_parser("enroll", help="注册声纹")
    p_reg.add_argument("speaker_id", help="说话人 ID")
    p_reg.add_argument("audio", help="音频文件路径")

    # ---- recognize ----
    p_rec = sub.add_parser("recognize", help="识别声纹")
    p_rec.add_argument("audio", help="音频文件路径")
    p_rec.add_argument(
        "speaker_ids", nargs="?", default=None,
        help="候选说话人 ID，逗号分隔（默认使用内置声纹）",
    )

    # ---- list-voiceprints ----
    sub.add_parser("list-voiceprints", help="列出所有内置声纹")

    args = parser.parse_args()

    client = WespeakerDeep(base_url=args.url, api_key=args.key)

    if args.cmd == "enroll":
        pk_path = f"voice_{args.speaker_id}.pkl"
        r = client.enroll(args.audio, pk_path)
        print(f"{'✅' if r['ok'] else '❌'} {r['msg']}")

    elif args.cmd == "recognize":
        if args.speaker_ids:
            client.load_templates(files={s: "" for s in args.speaker_ids.split(",")})
        else:
            client.load_templates(indices=[0])
        r = client.recognize(args.audio)
        status = "✅ 识别成功" if r["is_recognized"] else "❌ 未识别"
        print(f"{status}  name={r['name']}  confidence={r['confidence']:.4f}")

    elif args.cmd == "list-voiceprints":
        from ._voiceprints import _PEOPLE
        print("内置声纹列表:")
        for i, name in enumerate(_PEOPLE):
            print(f"  {i}: {name}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
