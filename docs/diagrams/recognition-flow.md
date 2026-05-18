# 识别流程详细图 (Detailed Recognition Flow)

## 完整识别流程 — WespeakerDeep

```mermaid
flowchart TD
    classDef input fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef config fill:#e0e0e0,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef compare fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef deep fill:#e0f2f1,stroke:#00796b,stroke-width:2px;

    Start([开始识别 Deep]):::input
    InputAudio[输入测试音频路径]:::input
    InputPk[参考声纹 .pkl 路径]:::input

    Config[DeepConfig 参数:<br/>sim_threshold: 0.50<br/>verify_crop_mode: head_window<br/>verify_buffer_keep_secs: 60.0<br/>enable_vad: False<br/>enable_score_compensation: True<br/>score_compensation_mode: sqrt<br/>enable_multi_template: True]:::config

    LoadModel[加载 ResNet34 模型<br/>pyannote.audio 3.3+]:::process

    LoadAudio[加载测试音频<br/>torchaudio.load / soundfile]:::process
    CheckFormat{文件格式?}:::decision
    ConvertM4a[转换 M4A → WAV<br/>ffmpeg]:::process
    ConvertError[错误: 不支持的格式]:::error

    Resample[重采样到 16kHz<br/>torchaudio.transforms.Resample]:::process
    ToMono[转为单声道<br/>waveform.mean]:::process

    GetDuration[获取音频时长]:::process
    CheckDuration{空/过短?}:::decision
    AudioTooShort[返回错误<br/>audio too short]:::error

    ApplyVAD{enable_vad = True?}:::decision
    VadProcess[Silero VAD 处理<br/>threshold: 0.5<br/>min_speech: 250ms]:::process
    SkipVAD[跳过 VAD<br/>使用完整音频]:::process

    HeadWindow[head_window 裁剪<br/>保留头部 N 秒<br/>verify_window_secs]:::process
    CropTail[tail 裁剪<br/>保留最后 N 秒<br/>默认不使用]:::process

    BufferCheck{时长 > buffer_keep?}:::decision
    Truncate[截断到 buffer_keep_secs<br/>保留头部 60 秒]:::process
    NoTruncate[保留完整音频]:::process

    IsShort{时长 < 1.5s?<br/>short_audio_max_duration}:::decision
    SlidingWindow[滑动窗口<br/>window: 0.4s, hop: 0.15s<br/>提取多个 window embedding]:::deep
    SingleEmb[单次 embedding 提取]:::process

    LoadFullPkl[加载完整 .pkl<br/>load_full() → templates 列表]:::process

    MultiTemplate[多模板匹配<br/>对所有 (window × template) 组合<br/>取 max 余弦相似度]:::deep
    SingleMatch[单模板匹配<br/>cosine(test, reference)]:::process

    ScoreComp["sqrt 分数补偿<br/>factor = (target/duration)^0.5<br/>comp = min(score × factor, 1.0)"]:::deep

    CompareThreshold{补偿后分数 >= 0.50?}:::decision

    IsRecognized[识别成功<br/>is_recognized = True]:::output
    NotRecognized[识别失败<br/>is_recognized = False]:::output

    BuildResult[构建返回结果]:::process
    Result["返回:<br/>is_recognized: bool<br/>confidence: float<br/>raw_confidence: float<br/>threshold: 0.50<br/>num_templates_used: int<br/>sliding_windows_used: int<br/>score_compensation_factor: float<br/>vad_duration: float"]:::output

    End([结束]):::output

    Start --> InputAudio
    Start --> InputPk

    Config -.->|参数| LoadModel
    Config -.->|参数| ApplyVAD
    Config -.->|参数| HeadWindow
    Config -.->|参数| BufferCheck
    Config -.->|参数| IsShort
    Config -.->|参数| SlidingWindow
    Config -.->|参数| MultiTemplate
    Config -.->|参数| ScoreComp
    Config -.->|参数| CompareThreshold

    LoadModel --> LoadAudio
    LoadModel --> SlidingWindow
    LoadModel --> SingleEmb

    LoadAudio --> CheckFormat
    CheckFormat -->|WAV| Resample
    CheckFormat -->|M4A| ConvertM4a
    CheckFormat -->|其他| ConvertError
    ConvertM4a --> Resample

    Resample --> ToMono
    ToMono --> GetDuration
    GetDuration --> CheckDuration
    CheckDuration -->|正常| ApplyVAD
    CheckDuration -->|空/过短| AudioTooShort

    ApplyVAD -->|True| VadProcess
    ApplyVAD -->|False| SkipVAD
    VadProcess --> HeadWindow
    SkipVAD --> HeadWindow

    HeadWindow --> BufferCheck
    BufferCheck -->|> buffer_keep| Truncate
    BufferCheck -->|<= buffer_keep| NoTruncate
    Truncate --> IsShort
    NoTruncate --> IsShort

    InputPk --> LoadFullPkl
    LoadFullPkl --> IsShort

    IsShort -->|是 短音频| SlidingWindow
    IsShort -->|否 长音频| SingleEmb

    SlidingWindow --> SlidingEmb[各窗口 embeddings]:::deep
    SlidingEmb --> MultiTemplate
    SingleEmb --> MultiTemplate

    MultiTemplate -->|raw_score| ScoreComp
    ScoreComp -->|compensated_score| CompareThreshold

    CompareThreshold -->|>= 0.50| IsRecognized
    CompareThreshold -->|< 0.50| NotRecognized

    IsRecognized --> BuildResult
    NotRecognized --> BuildResult
    BuildResult --> Result
    Result --> End
    AudioTooShort --> End
```

## 识别流程关键步骤

### 1. 音频加载与预处理

```
测试音频 (.wav/.m4a)
    ↓
格式转换 (M4A → WAV)
    ↓
重采样 → 16kHz
    ↓
单声道转换
    ↓
记录原始时长
```

### 2. VAD 处理（可选，默认关闭）

| 参数 | 值 | 说明 |
|------|------|------|
| threshold | 0.5 | VAD 置信度阈值 |
| min_speech_duration_ms | 250 | 最小语音时长 |
| min_silence_duration_ms | 100 | 最小静音时长 |

**注意**: `enable_vad = False` 是默认且实验验证的最佳选择。完整音频得分高于 VAD 去静音后的音频。

### 3. 音频裁剪策略

#### head_window（当前最优，DeepConfig 默认）
```
完整音频 → 检查时长
    ↓
超过 buffer_keep_secs?
    ↓ 是 → 保留头部 60 秒
    ↓ 否 → 保留完整音频
```

#### 旧版 full_utterance（WespeakerBest 默认）
```
完整音频 → 检查时长
    ↓
超过 buffer_keep_secs?
    ↓ 是 → 保留最后 60 秒
    ↓ 否 → 保留完整音频
```

### 4. 多模板匹配（核心改进）

```python
# 加载所有模板
full = load_full(pk_path)
templates = [normalize(t) for t in full["templates"]]

# 多模板 max 匹配
if enable_multi_template and num_templates > 1:
    scores = [dot(test_emb, t) for t in templates]
    raw_score = max(scores)
else:
    raw_score = dot(test_emb, reference)
```

**效果**: 多模板保持每文件独立 embedding，测试时对所有 template 取 max 分数，显著降低误拒绝率。

### 5. 短音频滑动窗口（可选，默认关闭）

短音频判定: `vad_duration < short_audio_max_duration (1.5s)`

```python
if enable_sliding_window_test and is_short:
    window_samples = int(0.4 * sample_rate)
    hop_samples = int(0.15 * sample_rate)
    for start in range(0, len - window + 1, hop):
        window = pcm[start:start + window]
        w_emb = extract_embedding(model, window)
        # 对每个 window 做多模板匹配
        for t in templates:
            score = dot(w_emb, t)
            best = max(best, score)
```

**默认关闭** (`enable_sliding_window_test = False`)，因为实验表明会推高 FAR。

### 6. sqrt 分数补偿

```python
mode = "sqrt"
target = 2.0  # score_compensation_target_duration
effective_dur = max(duration, 0.3)
factor = min((target / effective_dur) ** 0.5, 2.0)
compensated_score = min(raw_score * factor, 1.0)
```

**效果**: 短音频（如 0.5s）→ factor ~2.0，大幅提分；长音频（>= 2s）→ factor ~1.0，基本不变。

### 7. 阈值判断

| 阈值 | 值 | 说明 |
|------|------|------|
| **sim_threshold (DeepConfig)** | **0.50** | 18 轮实验验证的最优阈值 |
| 高安全 | 0.55-0.60 | 误接受率极低 |
| 高召回 | 0.40-0.45 | 误拒绝率极低 |

```
compensated_score >= 0.50?
    ↓ 是 → is_recognized = True
    ↓ 否 → is_recognized = False
```

### 8. 返回结果

```python
{
    "is_recognized": bool,            # 是否识别成功
    "confidence": float,              # 补偿后相似度 [0, 1]
    "raw_confidence": float,          # 原始相似度（补偿前）
    "threshold": 0.50,                # 使用的阈值
    "vad_duration": float,            # VAD 后时长（秒）
    "num_templates_used": int,        # 使用的模板数
    "sliding_windows_used": int,      # 滑动窗口数（0 = 未启用）
    "score_compensation_factor": float  # 分数补偿系数
}
```

## 识别流程参数配置

| 参数 | DeepConfig (当前最优) | BestConfig (旧版) |
|------|---------------------|-------------------|
| `sim_threshold` | **0.50** | 0.55 |
| `verify_crop_mode` | **head_window** | full_utterance |
| `verify_buffer_keep_secs` | 60.0 | 60.0 |
| `enable_vad` | False | False |
| `enable_score_compensation` | **True** | **False** |
| `score_compensation_mode` | **sqrt** | (linear) |
| `enable_multi_template` | **True** | False (不适用) |
| `enable_sliding_window_test` | False | False |
| `short_audio_max_duration` | 1.5s | N/A |

## 分数解释

| 分数范围 | 含义 | 说明 |
|---------|------|------|
| 0.70 - 1.00 | 非常匹配 | 极有可能是同一人 |
| 0.50 - 0.70 | 匹配 | 可能是同一人（阈值 0.50） |
| 0.40 - 0.50 | 模糊区 | 不确定，需要更多信息 |
| 0.30 - 0.40 | 不太匹配 | 可能不是同一人 |
| 0.00 - 0.30 | 不匹配 | 极可能不是同一人 |
| < 0.00 | 完全相反 | 不应该出现（检查错误） |
