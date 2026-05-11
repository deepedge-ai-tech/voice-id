# CLAUDE.md

## 项目概述

WeSpeaker 声纹识别工具 — 独立的声纹注册与识别 CLI 工具。
基于 pyannote.audio 后端，支持从音频文件提取声纹 embedding 并保存为 `.pkl`，后续通过余弦相似度进行说话人验证。

## 技术栈

- **语言**: Python 3
- **深度学习**: PyTorch, torchaudio
- **声纹模型**: pyannote.audio (ResNet34)
- **数值计算**: numpy
- **噪声增强**: audiomentations (可选)

## 项目结构

```
wespeaker/
├── wespeaker.py          # 主程序 (WespeakerClient 类 + CLI 入口)
├── clean.wav             # 音频素材
├── models/wespeaker/     # 预训练模型目录
├── voice.pkl             # 注册声纹输出
└── CLAUDE.md
```

## 遵循规范

本项目遵循公司 [Python 开发标准](../company-standards/)
- **强制使用**: company-global-constraints（全局 Plan 约束）
- **强制使用**: company-init（项目初始化约束）

## 常用命令

```bash
# 注册声纹
python wespeaker.py enroll audio.wav voice.pkl

# 识别声纹
python wespeaker.py recognize audio.wav voice.pkl

# 指定模型路径和设备
python wespeaker.py enroll audio.wav voice.pkl --model-path ./models/wespeaker --device cpu
```

## 核心 API

| 方法 | 说明 |
|------|------|
| `client.mp3_to_pk(audio_path, pk_path)` | 注册声纹，提取 embedding 并保存到 .pkl |
| `client.recognize(audio_path, pk_path)` | 比对音频与参考声纹，返回识别结果和置信度 |

## 强制检查清单

每个任务完成后必须验证：
- [ ] 代码使用 snake_case 命名
- [ ] 包含类型注解
- [ ] 使用 logging 而非 print（CLI 入口除外）
- [ ] 异常处理正确
- [ ] 无未使用的 import
