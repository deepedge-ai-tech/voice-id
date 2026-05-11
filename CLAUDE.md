# CLAUDE.md

## 项目概述

WeSpeaker 声纹识别工具 — 独立的声纹注册与识别 CLI 工具。
基于 pyannote.audio 后端，支持从音频文件提取声纹 embedding 并保存为 `.pkl`，后续通过余弦相似度进行说话人验证。

## 技术栈

- **语言**: Python 3.12+
- **包管理**: uv
- **深度学习**: PyTorch, torchaudio
- **声纹模型**: pyannote.audio (ResNet34)
- **数值计算**: numpy
- **噪声增强**: audiomentations (可选)
- **测试**: pytest + pytest-cov
- **格式化**: black + isort

## 遵循规范

本项目遵循公司 [Python 开发标准](../company-standards/)
- **强制使用**: company-global-constraints（全局 Plan 约束）
- **强制使用**: company-init（项目初始化约束）

## 项目结构

```
wespeaker/
├── src/                          # 源代码
│   └── wespeaker/
│       ├── __init__.py
│       └── wespeaker.py          # WespeakerClient 核心类
├── tests/                        # 测试（与 src 结构对应）
│   └── wespeaker/
│       ├── __init__.py
│       ├── conftest.py           # 共享 fixtures
│       └── test_wespeaker.py     # 核心功能测试
├── scripts/                      # 可执行脚本
│   ├── best_recognition.py       # 最佳配置注册与识别（multi-SNR 噪声注入）
│   ├── experiment_noise_optimization.py    # 四种优化方案对比实验
│   ├── experiment_train_noise_injection.py # 训练阶段噪声注入实验
│   ├── split_registration.py     # 按静音间隔切分注册音频
│   └── test_sliding_window.py    # 滑动窗口对比测试
├── docs/                         # 文档
│   ├── diagrams/                 # 项目图表（6 种 Mermaid 图）
│   │   ├── architecture.md
│   │   ├── data-flow.md
│   │   ├── sequence.md
│   │   ├── modules.md
│   │   ├── tech-stack.md
│   │   └── roadmap.md
│   └── test-plan.md              # 测试方案文档
├── asset/                        # 音频素材（不提交到 git）
│   └── john/
│       ├── registration_segments/ # 注册片段
│       ├── test_clean_segments/   # 安静环境测试
│       └── test_noise_segments/   # 嘈杂环境测试
├── models/                       # 预训练模型（不提交到 git）
├── pyproject.toml               # 项目配置
├── uv.lock                       # 锁定依赖
├── .gitignore
├── CLAUDE.md
├── AGENT.md
└── README.md
```

## 常用命令

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest

# 检查覆盖率
uv run pytest --cov --cov-fail-under=80

# 代码格式化
uv run black . && uv run isort .

# 提交前检查
uv run pytest --cov && uv run black . && uv run isort .

# 注册声纹
uv run python -m src.wespeaker.wespeaker enroll audio.wav voice.pkl

# 识别声纹
uv run python -m src.wespeaker.wespeaker recognize audio.wav voice.pkl

# 切分注册音频
uv run python scripts/split_registration.py asset/john/注册.aif asset/john/registration_segments

# 滑动窗口测试
uv run python scripts/test_sliding_window.py

# 最佳配置注册与识别
uv run python scripts/best_recognition.py enroll \
    --clean asset/john/registration_segments/ \
    --noise asset/john/test_noise_segments/嘈杂环境测试.m4a \
    --output asset/john/voice_best.pkl
uv run python scripts/best_recognition.py recognize \
    --audio asset/john/test_clean_segments/安静环境测试测试.m4a \
    --voiceprint asset/john/voice_best.pkl

# 实时声纹监控
uv run python -m wespeaker.realtime_monitor --voiceprint asset/john/voice_best.pkl
uv run python -m wespeaker.realtime_monitor --list-devices
```

## 最佳配置（基于实验结果）

| 参数 | 值 | 来源 |
|------|------|------|
| sim_threshold | **0.55** | 动态阈值分析 — clean 通过率 98.3% |
| verify_crop_mode | **full_utterance** | full utterance 得分始终高于滑动窗口 |
| verify_buffer_keep_secs | **60.0** | 不截断，使用完整音频 |
| enable_vad | **False** | 实验表明完整音频得分高于 VAD 去静音后 |
| 注册增强 | **multi-SNR 真实噪声注入** | 最优方案 — noisy +0.058, clean -0.038 |
| SNR 级别 | 20, 15, 10, 5, 0 dB | 多级别覆盖不同噪声强度 |

## 核心 API

| 方法 | 说明 |
|------|------|
| `WespeakerClient().enroll(audio_path, pk_path)` | 注册声纹，提取 embedding 并保存到 .pkl |
| `WespeakerClient().recognize(audio_path, pk_path)` | 比对音频与参考声纹，返回识别结果和置信度 |

## 强制检查清单

每个任务完成后必须验证：
- [ ] 代码符合命名规范（文件名 kebab-case，函数/变量 snake_case，类 PascalCase）
- [ ] 函数包含类型注解和 docstring
- [ ] 测试已编写或更新，pytest 全部通过
- [ ] 测试覆盖率 ≥ 80%
- [ ] 代码已格式化（black + isort）
- [ ] 不使用 print()（使用 logging）
- [ ] 不使用裸 except
- [ ] 项目图表已更新（6 种 Mermaid 图）
