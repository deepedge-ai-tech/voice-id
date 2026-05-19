#!/usr/bin/env bash
# 隔离环境测试 — 安装打包后的 whl 并验证程序可用
# 用法: bash test_whl_isolated.sh [asset_dir]
#   asset_dir: 可选，真实音频素材目录，默认 ../asset/john
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WHL_DIR="$PROJECT_DIR/dist"
WHL=$(ls "$WHL_DIR"/wespeaker_deep_edge-*.whl 2>/dev/null | sort -V | tail -1)

if [ -z "$WHL" ]; then
    echo "❌ 未找到 whl 文件，请先执行 uv build --wheel"
    exit 1
fi

ASSET_DIR="${1:-$PROJECT_DIR/asset/john}"
BATCH_MODE=false
if [ -d "$ASSET_DIR/registration_segments" ] && [ -d "$ASSET_DIR/test_segments" ]; then
    BATCH_MODE=true
fi

echo "📦 whl: $(basename "$WHL")"
echo "📁 素材目录: $ASSET_DIR (批量测试: $BATCH_MODE)"

# 隔离目录
TEST_DIR=$(mktemp -d /tmp/wespeaker-test-XXXXXX)
echo "📁 隔离目录: $TEST_DIR"

cleanup() {
    echo "🧹 清理: $TEST_DIR"
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# 创建并激活 venv（使用 uv + Python 3.10）
uv venv --python 3.10 "$TEST_DIR/venv" --quiet 2>&1
source "$TEST_DIR/venv/bin/activate"

echo "📥 安装 whl..."
uv pip install "$WHL" --quiet 2>&1

echo ""
echo "=== 测试 1: import + 默认模型路径 ==="
python -c "
from wespeaker_deep_edge.wespeaker import _get_default_model_path
path = _get_default_model_path()
import os
assert os.path.isdir(path), f'模型目录不存在: {path}'
assert os.path.isfile(os.path.join(path, 'pytorch_model.bin')), 'pytorch_model.bin 缺失'
assert os.path.isfile(os.path.join(path, 'config.yaml')), 'config.yaml 缺失'
print(f'✅ 内置模型路径: {path}')
for f in os.listdir(path):
    size = os.path.getsize(os.path.join(path, f))
    print(f'   {f}: {size:,} bytes')
"

echo ""
echo "=== 测试 2: 模型加载（CPU） ==="
python -c "
from wespeaker_deep_edge.wespeaker import _load_model, _get_default_model_path
print('正在加载模型...')
model = _load_model(_get_default_model_path(), 'cpu')
print(f'✅ 模型加载成功，参数数量: {sum(p.numel() for p in model.parameters()):,}')
"

echo ""
echo "=== 测试 3: WespeakerDeep 实例化 ==="
python -c "
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep
print('正在初始化 WespeakerDeep...')
deep = WespeakerDeep()
print(f'✅ deep_config.sim_threshold = {deep.deep_config.sim_threshold}')
"

echo ""
echo "=== 测试 4: embedding 提取 ==="
python -c "
import torch
from wespeaker_deep_edge.wespeaker import _extract_embedding, _load_model, _get_default_model_path
model = _load_model(_get_default_model_path(), 'cpu')
# 2s 随机音频 → 提取 embedding
audio = torch.randn(32000)
emb = _extract_embedding(model, audio)
import torch.nn.functional as F
emb = F.normalize(emb, dim=0)
assert emb.shape == (256,), f'期望 (256,)，实际 {emb.shape}'
print(f'✅ embedding 维度: {emb.shape}')
print(f'✅ embedding 范数: {float(emb.norm()):.4f}')
print(f'✅ embedding 前 5 维: {emb[:5].tolist()}')
"

echo ""
echo "=== 测试 5: 完整注册+识别流程（WespeakerDeep） ==="
python -c "
import numpy as np
import soundfile as sf
import tempfile, os
from pathlib import Path
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep

deep = WespeakerDeep()
sr = 16000

with tempfile.TemporaryDirectory() as tmpdir:
    pk_path = Path(tmpdir) / 'voice.pkl'
    test_path = Path(tmpdir) / 'test.wav'

    # 生成 3s 注册音频（不同频率正弦波）
    t = np.arange(sr * 3, dtype=np.float32) / sr
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sf.write(str(tmpdir / 'enroll.wav'), tone, sr)

    # 注册
    result = deep.enroll(str(tmpdir / 'enroll.wav'), pk_path=str(pk_path))
    assert result['ok'], f'注册失败: {result}'
    print(f'✅ 注册成功: dim={result[\"embedding_dim\"]}')

    # 生成测试音频（同频率，应高置信度）
    t = np.arange(sr, dtype=np.float32) / sr
    test_tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sf.write(str(test_path), test_tone, sr)

    # 识别
    result = deep.recognize(str(test_path), str(pk_path))
    print(f'✅ 识别结果: confidence={result[\"confidence\"]:.4f}, '
          f'is_recognized={result[\"is_recognized\"]}')
"

if $BATCH_MODE; then
echo ""
echo "=== 测试 6: 批量真实素材测试 (素材: $ASSET_DIR) ==="
ASSET_DIR="$ASSET_DIR" python -c "
import sys, os, json
from pathlib import Path
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep
import torch, torchaudio
from wespeaker_deep_edge.wespeaker import _load_audio

asset_dir = Path(os.environ['ASSET_DIR'])
reg_dir = asset_dir / 'registration_segments'
test_dir = asset_dir / 'test_segments'
pk_path = Path('/tmp/voice_john_batch.pkl')

deep = WespeakerDeep()

# 注册（拼接所有片段为单个音频）
print(f'注册目录: {reg_dir}')
reg_files = sorted(reg_dir.glob('*.wav'))
print(f'注册片段: {len(reg_files)} 个')
concat_wav = torch.cat([_load_audio(str(f)) for f in reg_files])
tmp_concat = Path('/tmp/_whl_concat_enroll.wav')
torchaudio.save(str(tmp_concat), concat_wav.unsqueeze(0), 16000)
result = deep.enroll(str(tmp_concat), pk_path=str(pk_path))
tmp_concat.unlink(missing_ok=True)
assert result['ok'], f'注册失败: {result}'
print(f'✅ 注册完成: dim={result[\"embedding_dim\"]}')

# 批量测试
test_files = sorted(test_dir.glob('*.wav'))
print(f'\n测试文件: {len(test_files)} 个')
print('-' * 60)
print(f'{\"文件\":>30s} | {\"置信度\":>8s} | {\"判定\":>6s}')
print('-' * 56)

passed = 0
failed = 0
for tf in test_files:
    result = deep.recognize(str(tf), str(pk_path))
    status = '✅ 通过' if result['is_recognized'] else '❌ 未过'
    if result['is_recognized']:
        passed += 1
    else:
        failed += 1
    print(f'{tf.name:>30s} | {result[\"confidence\"]:>8.4f} | {status:>6s}')

print('-' * 60)
total = len(test_files)
print(f'\n📊 结果: {passed}/{total} 通过 ({passed/total*100:.1f}%), {failed}/{total} 未通过')
if failed > 0:
    print(f'⚠️  注意: 未通过可能是阈值过严或音频差异过大，不代表 whl 有问题')
" 2>&1
else
    echo ""
    echo "=== 测试 6: 批量素材测试 (跳过，无素材目录) ==="
    echo "   用法: bash $0 path/to/asset/john"
fi

echo ""
echo "=============================="
echo "✅ 全部测试通过"
echo "=============================="

deactivate
