# Voice-ID 声纹识别自动研究项目

> "One day, voice biometric research used to be done by meat computers in between coffee breaks. That era is long gone. Research is now entirely the domain of autonomous AI agents running voice recognition experiments across compute clusters. This repo is the story of how it all began for Voice-ID."

## 项目概述

这是一个基于 WeSpeaker/Pyannote.audio 的声纹识别自动研究项目。AI 代理将自动调整参数、运行实验、评估结果，并迭代优化，目标是：

1. **提高短音频（< 0.6s）的识别置信度**
2. **降低所有用户的错误识别（误接受）**
3. **降低所有用户的误拒绝率**

**注册方式**: 使用合并音频注册 — 将每个说话人的所有注册片段合并成一条音频进行注册，提高声纹的稳定性和代表性。

## 核心参数（可调范围）

| 参数 | 默认值 | 可调范围 | 说明 |
|------|--------|----------|------|
| `sim_threshold` | 0.55 | 0.20 - 0.75 | 相似度识别阈值 |
| `verify_crop_mode` | "full_utterance" | "full_utterance" / "tail_window" / "head_window" | 验证音频裁剪模式 |
| `verify_buffer_keep_secs` | 8.0 | 2.0 - 60.0 | 验证 buffer 最大保留时长 |
| `verify_window_secs` | 1.0 | 0.3 - 3.0 | 滑动窗口/裁剪窗口长度 |
| `enrollment_segment_secs` | 1.0 | 0.5 - 3.0 | 注册时分段长度 |
| `enable_vad` | True | True / False | 是否启用 VAD 去静音 |
| `vad_rms_threshold` | 0.005 | 0.001 - 0.02 | VAD 能量阈值 |
| `noise_injection_snrs` | [20,15,10,5,0] | 自定义列表 | 注册噪声注入 SNR 级别 |

**固定参数（不修改）**:
- `enable_score_compensation`: True - 分数补偿功能固定关闭
- `score_compensation_target_duration`: 2.0 - 固定值

## 评估指标

每次实验运行固定时间预算（约 5-10 分钟），评估指标：

| 指标 | 说明 | 目标 |
|------|------|------|
| `far` | False Accept Rate (误接受率) | ↓ 降低 |
| `frr` | False Reject Rate (误拒绝率) | ↓ 降低 |
| `short_audio_conf` | 短音频平均置信度（所有样本） | ↑ 提高 |
| `short_audio_genuine_conf` | 短音频同人平均置信度 | ↑ 提高 |
| `overall_accuracy` | 总体准确率 | ↑ 提高 |
| `eer` | Equal Error Rate | ↓ 降低 |

**短音频定义**（取决于 VAD 状态）：
- VAD 启用时：< 0.6s（VAD 裁掉静音后音频变短）
- VAD 禁用时：< 1.5s（原始音频长度，包含短音频场景）

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 准备数据（一次性）
uv run prepare.py

# 3. 手动运行单次实验（测试）
uv run training.py

# 4. 让 AI 代理开始自动研究
# 只需告诉代理: "看一下 program.md 并开始实验吧！"
```

## 研究指南

### 推荐探索方向

1. **短音频优化**（VAD 禁用时 < 1.5s，VAD 启用时 < 0.6s）
   - 降低 `sim_threshold`（针对短音频，如 0.30-0.40）
   - 减小 `verify_window_secs` 到 0.3-0.5
   - 设置 `verify_crop_mode` 为 "head_window"
   - 禁用 `enable_vad`（保留更多音频信息）

2. **降低误接受 (FAR)**
   - 提高 `sim_threshold`（如 0.60-0.70）
   - 启用 `enable_vad`
   - 降低 `vad_rms_threshold`（更严格去噪）
   - 使用 `full_utterance` 模式

3. **降低误拒绝 (FRR)**
   - 降低 `sim_threshold`（如 0.35-0.45）
   - 禁用 `enable_vad`（保留更多信息）
   - 禁止使用 `asset/john/嘈杂环境测试.m4a`,可以使用生成的噪声，或者使用VAD提取test集合中的噪声，提高鲁棒性
   - 增加 `noise_injection_snrs` 覆盖范围

4. **噪声鲁棒性**
   - 扩展 `noise_injection_snrs` 范围：[30,25,20,15,10,5,0,-5]
   - 降低 `enrollment_segment_secs` 以增加片段数量
   - 调整 `verify_buffer_keep_secs` 保留更多信息

### 参数组合建议

```python
# 短音频优先配置
{
    "sim_threshold": 0.35,
    "verify_crop_mode": "head_window",
    "verify_window_secs": 0.4,
    "enable_vad": False,
    "noise_injection_snrs": [20,15,10,5,0]
}

# 平衡配置
{
    "sim_threshold": 0.45,
    "verify_crop_mode": "full_utterance",
    "enable_vad": True,
    "vad_rms_threshold": 0.003,
    "noise_injection_snrs": [20,15,10,5,0]
}

# 高安全性配置（低 FAR）
{
    "sim_threshold": 0.65,
    "verify_crop_mode": "full_utterance",
    "enable_vad": True,
    "vad_rms_threshold": 0.008,
    "verify_buffer_keep_secs": 4.0
}
```

## 输出结构

每次实验结果保存到 `outputs/experiments/`:

```
outputs/
├── experiments/
│   ├── exp_0001_20260513_120000/
│   │   ├── config.json          # 实验参数配置
│   │   ├── metrics.json         # 评估指标
│   │   ├── heatmap.png          # 相似度热力图
│   │   └── summary.md           # 实验摘要
│   ├── exp_0002_20260513_120530/
│   └── ...
├── best_config.json             # 当前最佳配置
└── experiment_log.json          # 所有实验历史
```

## 代理工作流程

1. **分析上次实验结果** → 读取 `outputs/experiment_log.json`
2. **确定调整方向** → 基于 FAR/FRR/短音频置信度
3. **生成新配置** → 调整 training.py 中的参数
4. **运行实验** → `uv run training.py`
5. **评估结果** → 检查指标是否改善
6. **记录并迭代** → 保存结果，返回步骤 1

## 成功标准

- 必须达成：同人平均置信度 > 0.7
- 必须达成：短音频（VAD启用<0.6s / VAD禁用<1.5s）同人平均置信度 > 0.55
- FAR < 5%
- FRR < 10%
- EER < 8%

## 注意事项

- 每个 SPEAKERS 配置组合测试时间约 5-10 分钟
- 预计每小时可运行 6-12 次实验
- 重点关注 `outputs/best_config.json` 的演变
- 如果 同人平均置信度 没有达成则不要停止

---

**代理提示**: 你只需要修改 `training.py` 中的参数配置部分。不要修改 `prepare.py` — 它包含固定常量和工具函数。实验运行后会自动生成评估报告和可视化图表。
