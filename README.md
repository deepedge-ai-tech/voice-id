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
from wespeaker_deep_edge import WespeakerClient

# 初始化客户端
client = WespeakerClient()

# 注册声纹
client.mp3_to_pk("voice.mp3", "voice.pkl")

# 识别声纹（使用内置 John 声纹）
result = client.recognize("test.mp3")  # pk_path 默认 None

# 使用内置声纹（按 index）
client.package_pk_index = 1  # Frank
result = client.recognize("test.mp3")

# 指定外部声纹文件
result = client.recognize("test.mp3", "voice.pkl")
print(f"识别结果: {result}")
```

### 使用最佳配置（推荐）

```python
from wespeaker_deep_edge.wespeaker_deep_dege import WespeakerDeep

# 默认参数即为 18 轮实验验证的最优配置，无需额外设置
recognizer = WespeakerDeep(model_path="./models/wespeaker")

# 注册声纹（纯干净注册 + 多模板）
recognizer.enroll(
    clean_dir="registration_segments/",
    pk_path="voice.pkl"
)

# 识别
result = recognizer.recognize("test_audio.wav", "voice.pkl")
```

### 使用旧版最佳配置

```python
from wespeaker_deep_edge.best import WespeakerBest, BestConfig

recognizer = WespeakerBest(model_path="./models/wespeaker")

# 提取噪声 profile
noise_profile = WespeakerBest.extract_noise_profile("noise.wav")

# 注册（multi-SNR 噪声注入）
recognizer.enroll(
    clean_dir="registration_segments/",
    noise_profile=noise_profile,
    pk_path="voice.pkl"
)

result = recognizer.recognize("test_audio.wav", "voice.pkl")
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

### WespeakerBest（旧方案，供参考）

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

- Python 3.10+
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
