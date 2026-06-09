# WeSpeaker

WeSpeaker 声纹识别工具 — 独立的声纹注册与识别 CLI 工具。

支持 PyTorch (pyannote.audio) 和 ONNX Runtime 双引擎，从音频文件提取声纹 embedding 并保存为 `.pkl`，后续通过余弦相似度进行说话人验证。

## 特性

- 双引擎支持: PyTorch (pyannote.audio) + ONNX Runtime（CPU/GPU）
- 零项目依赖，仅需安装: torch, torchaudio, numpy, audiomentations(可选)
- 支持多种音频格式: wav, mp3, m4a, ogg
- 自动噪声增强，提高鲁棒性
- 提供最佳配置（基于 18 轮实验验证）
- CLI 工具 + Python API + WebSocket 服务端，开箱即用
- 内置 9 人声纹，无需外部文件即可识别

## 安装

### 从 PyPI（已发布）

```bash
# 默认安装（仅核心依赖: numpy, scipy, soundfile）
pip install wespeaker-deep-edge

# 完整安装（含 PyTorch 引擎 + 所有依赖）
pip install "wespeaker-deep-edge[deps]"

# ONNX CPU 引擎
pip install "wespeaker-deep-edge[cpu]"

# ONNX GPU 引擎（Jetson / CUDA）
pip install "wespeaker-deep-edge[gpu]"

# 组合安装
pip install "wespeaker-deep-edge[deps,cpu]"
```

### 从 GitHub 直接安装

```bash
# 默认安装（仅包本身，不装任何依赖）
pip install git+https://github.com/deepedge-ai-tech/voice-id.git

# 完整安装（含所有依赖）
pip install "wespeaker-deep-edge[deps] @ git+https://github.com/deepedge-ai-tech/voice-id.git"

# 指定版本 tag
pip install git+https://github.com/deepedge-ai-tech/voice-id.git@v0.1.27

# 或使用 uv
uv pip install git+https://github.com/deepedge-ai-tech/voice-id.git
```

### 从源码安装

```bash
git clone https://github.com/deepedge-ai-tech/voice-id.git
cd voice-id
pip install .           # 默认不装依赖
pip install ".[deps]"   # 完整 PyTorch 引擎安装
pip install ".[cpu]"    # ONNX CPU 引擎
pip install ".[gpu]"    # ONNX GPU 引擎
```

### 查看版本

```bash
python -m wespeaker_deep_edge --version
```

## 快速开始

### 命令行使用

```bash
# 注册声纹
wespeaker-deep-edge enroll voice.mp3 voice.pkl

# 识别声纹（使用内置 John 声纹，无需传 voiceprint）
wespeaker-deep-edge recognize test.mp3

# 指定内置声纹（按 index）
wespeaker-deep-edge recognize test.mp3 --package-pk-index 1   # Frank
wespeaker-deep-edge recognize test.mp3 --package-pk-index 3   # Qingqing
wespeaker-deep-edge recognize test.mp3 --package-pk-index 8   # John (Double Mic)

# 指定外部声纹文件
wespeaker-deep-edge recognize test.mp3 voice.pkl

# 同时传入 voiceprint 和 --package-pk-index：优先使用内置声纹
wespeaker-deep-edge recognize test.mp3 voice.pkl --package-pk-index 2   # Michael
```

### 内置声纹索引

| Index | Name              |
|-------|-------------------|
| 0     | John              |
| 1     | Frank             |
| 2     | Michael           |
| 3     | Qingqing          |
| 4     | Xixi              |
| 5     | Zhong             |
| 6     | Angle             |
| 7     | Albert            |
| 8     | John (Double Mic) |

> 不传 `voiceprint` 参数时，默认使用 John (index 0) 的声纹。

### Python API 使用

```python
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep, DeepConfig

# 默认参数即为 18 轮实验验证的最优配置
recognizer = WespeakerDeep()

# 注册声纹
recognizer.enroll("audio.wav", "voice.pkl")

# 识别声纹（使用内置 John 声纹，无需传 pkl）
result = recognizer.recognize("test.wav")
print(f"识别结果: {result}")

# 使用内置声纹（按 index）
cfg = DeepConfig(package_pk_index=1)  # Frank
recognizer2 = WespeakerDeep(config=cfg)
result = recognizer2.recognize("test.wav")

# 指定外部声纹文件
result = recognizer.recognize("test.wav", "voice.pkl")
```

## 最佳配置参数

以下配置经 18 轮自动实验验证并通过 `cross_test_merged.py` 交叉测试确认。

### WespeakerDeep（推荐，当前最优）

| 参数 | 值 | 说明 |
|------|------|------|
| sim_threshold | **0.50** | 余弦相似度阈值 |
| verify_crop_mode | head_window | 超长音频保留头部 |
| verify_buffer_keep_secs | 60.0 | 不截断音频 |
| verify_window_secs | 0.4 | 短音频滑动窗口长度 |
| enrollment_segment_secs | 0.6 | 注册片段长度 |
| enable_vad | False | 完整音频得分更高 |
| enable_score_compensation | True | sqrt 补偿短音频分数 |
| score_compensation_mode | sqrt | sqrt 模式补偿 |
| score_compensation_target_duration | 2.0 | 补偿目标时长 |
| enroll_skip_vad | True | 注册时跳过 VAD |
| enroll_clean_only | True | 纯净注册，不注入噪声 |
| enable_multi_template | True | 多模板匹配（取 max） |
| enable_sliding_window_test | False | 短音频滑动窗口（默认关闭，避免推高 FAR） |
| short_audio_max_duration | 1.5 | 短音频判定阈值（秒） |
| noise_injection_snrs | () | 无噪声注入 |

**注册流程**: 纯干净注册 → 每文件独立 embedding → 多模板保存
**识别流程**: 多模板 max 匹配 → sqrt 分数补偿 → 短音频自动提分

## 引擎选择

| 引擎 | 后端 | 适用场景 |
|------|------|----------|
| `WespeakerDeep` (默认) | PyTorch / pyannote.audio | 开发、训练、完整功能 |
| `OnnxEngine` | ONNX Runtime | 边缘部署 (Jetson)、低资源设备 |

```python
# PyTorch 引擎（默认）
from wespeaker_deep_edge import WespeakerDeep
client = WespeakerDeep()

# ONNX 引擎
from wespeaker_deep_edge.onnx_engine import OnnxEngine, OnnxConfig
cfg = OnnxConfig(onnx_model_dir="/path/to/models")
client = OnnxEngine(config=cfg)
```

## 项目结构

```
Voice-ID/
├── pyproject.toml                    # 项目配置与依赖
├── src/wespeaker_deep_edge/          # ★ 主源码包
│   ├── wespeaker_deep_dege.py        #   核心引擎 (WespeakerDeep, PyTorch)
│   ├── onnx_engine.py                #   ONNX 推理引擎 (OnnxEngine)
│   ├── asnorm.py                     #   Adaptive S-Norm 分数归一化
│   ├── _utils.py                     #   通用工具函数
│   ├── server/                       #   WebSocket 服务端
│   │   ├── ws_server.py              #     WS 协议处理
│   │   └── template_manager.py       #     多模板矩阵管理
│   ├── client/                       #   WebSocket 客户端 SDK
│   │   └── speaker_client.py         #     SpeakerClient
│   ├── _voiceprints/                 #   内置声纹（9人）
│   ├── _models/                      #   内置模型 (PyTorch + ONNX)
│   ├── _cohort/                      #   队列数据 (AS-Norm)
│   └── _wespeaker/                   #   vendored 官方 WeSpeaker
├── tests/                            # 测试套件
├── scripts/                          # 实用脚本
└── docs/                             # 文档
```

## 依赖

### 核心依赖（默认安装）

- Python 3.10+
- numpy == 1.26.4
- scipy >= 1.0
- soundfile >= 0.13.1

### PyTorch 引擎（`[deps]` 可选安装）

- torch >= 2.8.0
- torchaudio >= 2.8.0
- pyannote-audio >= 3.3.2
- silero-vad
- audiomentations >= 0.43.1

### ONNX 引擎（`[cpu]` / `[gpu]` 可选安装）

- onnxruntime >= 1.16.0（CPU）
- onnxruntime-gpu（GPU / Jetson）

## 开发

```bash
# 安装开发依赖
uv sync --group dev

# 运行测试
pytest

# 代码格式化
black . && isort .
```

## 许可证

MIT License
