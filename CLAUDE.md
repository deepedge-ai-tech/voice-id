# CLAUDE.md

## 项目概述

WeSpeaker 声纹识别工具 — 独立的声纹注册与识别 CLI 工具。
基于 pyannote.audio 后端，支持从音频文件提取声纹 embedding 并保存为 `.pkl`，后续通过余弦相似度进行说话人验证。

## 技术栈

- **语言**: Python 3.10+
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
├── src/wespeaker_deep_edge/       # 源代码
│   ├── __init__.py
│   ├── __main__.py                # CLI 入口
│   ├── wespeaker.py               # WespeakerClient 核心类
│   ├── best.py                    # WespeakerBest（旧方案，multi-SNR 噪声注入）
│   ├── wespeaker_deep_dege.py     # WespeakerDeep（当前最优方案）
│   ├── diagnostics.py             # 诊断工具
│   ├── reporters.py               # 报告生成
│   ├── realtime_monitor.py        # 实时监控
│   └── _models/wespeaker/         # 内置预训练模型（打包进 whl）
│       ├── pytorch_model.bin      # ResNet34 权重 (25MB)
│       └── config.yaml            # 模型配置
├── tests/                         # 测试
│   ├── wespeaker/                 # WespeakerClient + Best 测试
│   └── wespeaker_deep_edge/       # WespeakerDeep 测试
├── scripts/                       # 可执行脚本
│   ├── cross_test_merged.py       # 交叉测试（WespeakerDeep，验证最佳配置）
│   ├── test_whl_isolated.sh       # whl 隔离环境测试（打包前必跑）
│   ├── best_recognition.py        # 最佳配置注册与识别
│   ├── split_registration.py      # 按静音间隔切分注册音频
│   └── test_sliding_window.py     # 滑动窗口对比测试
├── asset/                         # 音频素材（不提交到 git）
├── models/                        # 符号链接 → HuggingFace 缓存（开发用，不打包）
├── dist/                          # whl 产物（不提交到 git）
├── pyproject.toml
├── CLAUDE.md
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

# 构建 whl（含内置模型，打包前必须先跑单元测试 + 隔离测试）
uv run pytest && bash scripts/test_whl_isolated.sh && uv build --wheel

# 隔离环境测试（安装 whl → 全套验证，支持传入素材目录）
bash scripts/test_whl_isolated.sh [asset_dir]

# 注册声纹
uv run python -m wespeaker_deep_edge.wespeaker enroll audio.wav voice.pkl

# 识别声纹（不传 voiceprint 默认用内置 John 声纹）
uv run python -m wespeaker_deep_edge.wespeaker recognize audio.wav

# 识别声纹（指定内置声纹 index）
uv run python -m wespeaker_deep_edge.wespeaker recognize audio.wav --package-pk-index 1  # Frank

# 识别声纹（指定外部文件）
uv run python -m wespeaker_deep_edge.wespeaker recognize audio.wav voice.pkl

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

## 最佳配置（18 轮实验验证 + cross_test_merged.py 交叉测试确认）

全部为 `WespeakerDeep` / `DeepConfig` 默认值，无需额外设置。

| 参数 | 值 | 说明 |
|------|------|------|
| sim_threshold | **0.50** | 余弦相似度阈值 |
| verify_crop_mode | head_window | 超长音频保留头部 |
| verify_buffer_keep_secs | 60.0 | 不截断音频 |
| enable_vad | False | 完整音频得分更高 |
| enable_score_compensation | True | sqrt 补偿短音频分数 |
| enroll_skip_vad | True | 注册跳过 VAD |
| enroll_clean_only | True | 纯干净注册，不注入噪声 |
| enable_multi_template | True | 多模板匹配（取 max） |
| enable_sliding_window_test | False | 短音频滑动窗口（默认关闭） |

**注册流程**: 纯净注册 → 每文件独立 embedding → 多模板保存
**识别流程**: 多模板 max 匹配 → sqrt 分数补偿 → 短音频自动提分

## 调试音频自动保存

每次 `recognize()` 调用时，输入的音频会自动保存到系统临时目录的 `wespeaker_debug/` 文件夹：

```bash
# 查看保存位置
ls $(python3 -c "import tempfile; print(tempfile.gettempdir())")/wespeaker_debug/
```

文件名格式: `{日期时间}-{置信度}.wav`，无需任何环境变量。

## 内置声纹

6 人声纹已打包进 whl（`_voiceprints/`），CLI/Python API 均可使用。

| Index | Name     |
|-------|----------|
| 0     | John     |
| 1     | Frank    |
| 2     | Michael  |
| 3     | Qingqing |
| 4     | Xixi     |
| 5     | Zhong    |

- CLI: `--package-pk-index <index>` 选择内置声纹，不传 `voiceprint` 时默认 John (index 0)
- Python API: 设置 `client.package_pk_index` 或 `recognize()` 不传 `pk_path`
- `package_pk_index` 优先级高于 `pk_path`

## 核心 API

| 方法 | 说明 |
|------|------|
| `WespeakerClient().enroll(audio_path, pk_path)` | 注册声纹，提取 embedding 并保存到 .pkl |
| `WespeakerClient().recognize(audio_path, pk_path=None)` | 比对音频与参考声纹，返回识别结果和置信度。**pk_path=None 时使用内置声纹**。调试音频自动保存 |

## 强制检查清单

每个任务完成后必须验证：
- [ ] 代码符合命名规范（文件名 kebab-case，函数/变量 snake_case，类 PascalCase）
- [ ] 函数包含类型注解和 docstring
- [ ] 测试已编写或更新，pytest 全部通过
- [ ] 测试覆盖率 ≥ 30%（pyproject.toml 阈值）
- [ ] 代码已格式化（black + isort）
- [ ] 不使用 print()（使用 logging）
- [ ] 不使用裸 except

## 发布流程

**顺序执行，任何一步失败则停止：**

1. **单元测试**: `uv run pytest`
2. **隔离测试**: `bash scripts/test_whl_isolated.sh`
3. **构建 whl**: `uv build --wheel`
4. **打 tag 并推送**: `git tag v0.1.x && git push origin v0.1.x`

隔离测试会创建独立 venv → 安装 whl → 验证模型加载、embedding 提取、注册+识别全流程，以及真实素材批量测试（13/16 通过基准）。
