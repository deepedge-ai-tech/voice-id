# WeSpeaker

WeSpeaker 声纹识别工具 — 独立的声纹注册与识别 CLI 工具。

基于 pyannote.audio 后端，支持从音频文件提取声纹 embedding 并保存为 `.pkl`，后续通过余弦相似度进行说话人验证。

## 特性

- 零项目依赖，仅需安装: torch, torchaudio, numpy, audiomentations(可选)
- 支持多种音频格式: wav, mp3, m4a, ogg
- 自动噪声增强，提高鲁棒性
- 提供最佳配置（基于实验验证）
- CLI 工具，开箱即用

## 安装

```bash
# 从源码安装（完整安装，含服务端推理）
pip install .

# 或使用 uv
uv pip install .

# 仅安装客户端 SDK（轻量，无需 torch）
pip install "wespeaker-deep-edge[client]"

# 仅安装服务端
pip install "wespeaker-deep-edge[server]"
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

# 指定外部声纹文件
wespeaker-deep-edge recognize test.mp3 voice.pkl

# 同时传入 voiceprint 和 --package-pk-index：优先使用内置声纹
wespeaker-deep-edge recognize test.mp3 voice.pkl --package-pk-index 2   # Michael
```

### 内置声纹索引

| Index | Name     |
|-------|----------|
| 0     | John     |
| 1     | Frank    |
| 2     | Michael  |
| 3     | Qingqing |
| 4     | Xixi     |
| 5     | Zhong    |

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

### WebSocket 服务端

```bash
# 启动服务
wespeaker-deep-edge-server --port 10000 --storage-dir ./voiceprints
```

### WebSocket 客户端 SDK

```python
from wespeaker_deep_edge.client import SpeakerClient

client = SpeakerClient("ws://localhost:10000")
await client.connect()

# 注册声纹
await client.enroll("audio.wav", "user_001")

# 加载模板（支持预设声纹 preset_* 和用户注册 ID）
await client.load(["user_001", "preset_john", "preset_frank"])

# 识别（矩阵批量 cosine similarity，返回最高分）
result = await client.recognize("test.wav")
print(f"识别结果: {result}")  # {"id": "preset_john", "score": 0.8523}

await client.close()
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

## 项目结构

```
Voice-ID/
├── pyproject.toml                    # 项目配置与依赖
├── src/wespeaker_deep_edge/          # ★ 主源码包
│   ├── wespeaker_deep_dege.py        #   核心引擎 (WespeakerDeep)
│   ├── server/                       #   WebSocket 服务端
│   │   ├── ws_server.py              #     WS 协议处理
│   │   └── template_manager.py       #     多模板矩阵管理
│   ├── client/                       #   WebSocket 客户端 SDK
│   │   └── speaker_client.py         #     SpeakerClient
│   ├── _voiceprints/                 #   内置声纹（8人）
│   ├── _models/                      #   内置模型
│   └── _wespeaker/                   #   vendored 官方 WeSpeaker
├── tests/                            # 测试套件
├── scripts/                          # 实用脚本
└── docs/                             # 文档
```

## 依赖

- Python 3.10+
- torch >= 2.8.0
- torchaudio >= 2.8.0
- numpy == 1.26.4
- pyannote-audio >= 3.3.2
- websockets >= 12.0（server/client）
- silero-vad
- audiomentations >= 0.43.1（可选）

> client 安装仅需 websockets + numpy + soundfile，不含 torch。

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
