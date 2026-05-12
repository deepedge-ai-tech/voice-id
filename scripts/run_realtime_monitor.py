#!/usr/bin/env python3
"""实时声纹监控运行器 — 包装脚本，解决模块路径问题。

用法:
    uv run python scripts/run_realtime_monitor.py --list-devices
    uv run python scripts/run_realtime_monitor.py --voiceprint asset/john/voice_best.pkl
    uv run python scripts/run_realtime_monitor.py --window-secs 3.0 --step-secs 0.1
"""

import sys
from pathlib import Path

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wespeaker.realtime_monitor import main

if __name__ == "__main__":
    # 如果没有参数且默认 voiceprint 存在，使用默认值
    if len(sys.argv) == 1:
        default_vp = Path("asset/john/voice_best.pkl")
        if default_vp.exists():
            sys.argv.extend(["--voiceprint", str(default_vp)])
    main()
