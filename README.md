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
# 从源码安装
pip install .

# 或使用 uv
uv pip install .
```

## 快速开始

### 命令行使用

```bash
# 注册声纹
wespeaker enroll voice.mp3 voice.pkl

# 识别声纹
wespeaker recognize test.mp3 voice.pkl
```

### Python API 使用

```python
from wespeaker import WespeakerClient

# 初始化客户端
client = WespeakerClient()

# 注册声纹
client.mp3_to_pk("voice.mp3", "voice.pkl")

# 识别声纹
result = client.recognize("test.mp3", "voice.pkl")
print(f"识别结果: {result}")
```

### 使用最佳配置

```python
from wespeaker.best import WespeakerBest, BestConfig

# 初始化（使用默认最佳配置）
recognizer = WespeakerBest(model_path="./models/wespeaker")

# 提取噪声 profile（用于注册时噪声注入）
noise_profile = WespeakerBest.extract_noise_profile("noise.wav")

# 注册声纹（multi-SNR 真实噪声注入）
recognizer.enroll(
    clean_dir="registration_segments/",
    noise_profile=noise_profile,
    pk_path="voice.pkl"
)

# 识别
result = recognizer.recognize("test_audio.wav", "voice.pkl")
```

## 最佳配置参数

| 参数 | 值 | 说明 |
|------|------|------|
| sim_threshold | 0.55 | 识别阈值 |
| verify_crop_mode | full_utterance | 使用完整音频 |
| verify_buffer_keep_secs | 60.0 | 不截断音频 |
| enable_vad | False | 完整音频得分更高 |
| 注册增强 | multi-SNR 真实噪声注入 | 最优方案 |

## 项目结构

```
wespeaker/
├── src/wespeaker/
│   ├── __init__.py
│   ├── wespeaker.py      # 核心客户端
│   └── best.py           # 最佳配置
├── tests/                # 测试套件
├── scripts/              # 实用脚本
└── docs/                 # 文档
```

## 依赖

- Python 3.12+
- torch >= 2.11.0
- torchaudio >= 2.11.0
- numpy >= 2.4.4
- audiomentations >= 0.43.1
- pyannote-audio >= 4.0.4
- silero-vad >= 5.1.2

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
