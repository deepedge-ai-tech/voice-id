#!/usr/bin/env bash
# ── Jetson 服务端依赖安装脚本 ──────────────────────────────────────────────────
# 用法:
#   bash scripts/install_jetson_deps.sh              # uv 安装
#   bash scripts/install_jetson_deps.sh pip           # pip 安装
#
# 从 Jetson AI Lab (jp6/cu126) 安装 torch/torchaudio/torchvision 的 aarch64 wheel。
# 先安装这三个核心库，再安装项目其余依赖。
#
# 注意:
#   - 仅适用于 JetPack 6 / CUDA 12.6 / Python 3.10
#   - 需要预先安装 uv (pip install uv) 或 pip
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

TORCH_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
TORCHAUDIO_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=81a775c8af36ac859fb3f4a1b2f662d5fcf284a835b6bb4ed8d0827a6aa9c0b7"
TORCHVISION_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1"

echo "========================================"
echo " WeSpeaker - Jetson 服务端依赖安装"
echo "========================================"

if [ "${1:-}" = "pip" ]; then
    echo "[1/3] 安装 torch..."
    pip --no-cache install "$TORCH_URL"
    echo "[2/3] 安装 torchaudio..."
    pip --no-cache install "$TORCHAUDIO_URL"
    echo "[3/3] 安装 torchvision..."
    pip --no-cache install "$TORCHVISION_URL"
else
    echo "[1/3] 安装 torch..."
    uv add --no-cache "$TORCH_URL"
    echo "[2/3] 安装 torchaudio..."
    uv add --no-cache "$TORCHAUDIO_URL"
    echo "[3/3] 安装 torchvision..."
    uv add --no-cache "$TORCHVISION_URL"
fi

echo ""
echo "核心库安装完成。继续安装项目依赖..."

if [ "${1:-}" = "pip" ]; then
    pip install -e ".[server]"
else
    uv sync --extra server
fi

echo ""
echo "========================================"
echo " 全部完成！启动服务:"
echo " wespeaker-deep-edge-server --port 10000 --storage-dir ./voiceprints"
echo "========================================"
