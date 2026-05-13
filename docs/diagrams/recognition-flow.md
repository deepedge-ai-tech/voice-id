# 识别流程详细图 (Detailed Recognition Flow)

## 完整识别流程

```mermaid
flowchart TD
    classDef input fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef config fill:#e0e0e0,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef compare fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    Start([开始识别]):::input
    InputAudio[输入测试音频路径]:::input
    InputPk[参考声纹 .pkl 路径]:::input

    Config[配置参数:<br/>sim_threshold: 0.55<br/>verify_crop_mode: full_utterance<br/>verify_buffer_keep_secs: 60.0<br/>enable_vad: False<br/>enable_score_compensation: False]:::config

    LoadModel[加载 ResNet34 模型<br/>pyannote.audio 3.1]:::process

    LoadAudio[加载测试音频<br/>torchaudio.load]:::process
    CheckFormat{文件格式?}:::decision
    ConvertM4a[转换 M4A → WAV<br/>ffmpeg]:::process
    ConvertError[错误: 不支持的格式]:::error

    Resample[重采样到 16kHz<br/>torchaudio.transforms.Resample]:::process
    ToMono[转为单声道<br/>waveform.mean]:::process

    GetDuration[获取音频时长]:::process
    CheckDuration{时长检测}:::decision

    RecordInfo[记录原始信息<br/>original_duration]:::process

    ApplyVAD{enable_vad = True?}:::decision
    VadProcess[Silero VAD 处理<br/>threshold: 0.5<br/>min_speech: 250ms<br/>window_size: 512]:::process
    SkipVAD[跳过 VAD<br/>使用完整音频]:::process
    VadDone[VAD 处理完成<br/>vad_duration]:::output

    CheckCrop{verify_crop_mode}:::decision
    CropFull[full_utterance<br/>使用完整音频]:::process
    CropWindow[sliding_window<br/>滑动窗口]:::process

    BufferCheck{时长 > buffer_keep?}:::decision
    Truncate[截断到 buffer_keep_secs<br/>保留最后 N 秒]:::process
    NoTruncate[保留完整音频]:::process

    CroppedAudio[裁剪后音频]:::output
    RecordCrop[记录裁剪信息<br/>final_duration]:::process

    LoadRefPkl[加载参考声纹 .pkl<br/>pickle.load]:::process
    ValidatePkl{验证 pkl 格式}:::decision
    PklError[错误: 无效的 .pkl 文件]:::error

    RefEmbedding[参考 Embedding<br/>256 维向量]:::output
    CheckRefDim{维度 = 256?}:::decision
    RefError[错误: Embedding 维度错误]:::error

    CalcRMS[计算 RMS 能量<br/>rms = sqrt(mean(x²))]:::process
    RecordRMS[记录 rms_energy<br/>用于质量评估]:::process

    ExtractTest[提取测试 Embedding]:::process
    Preprocess[预处理<br/>归一化 [-1, 1]]:::process
    ModelForward[模型前向传播<br/>ResNet34 → 256 维向量]:::process
    TestNorm[L2 归一化<br/>||v|| = 1.0]:::process
    TestEmbedding[测试 Embedding<br/>256 维向量]:::output

    ComputeSim[计算余弦相似度<br/>cosine(ref, test)<br/>= dot(ref, test)]:::compare

    ScoreCompensation{enable_score_compensation?}:::decision
    GetVadDuration[获取 VAD 后时长<br/>vad_duration]:::process
    CalcCompensation[计算补偿系数<br/>factor = target / vad_duration<br/>如果 vad < target]:::process
    ApplyCompensation[应用补偿<br/>adjusted_score = score × factor]:::process
    NoCompensation[使用原始分数]:::process

    FinalScore[最终相似度分数<br/>confidence]:::output

    CompareThreshold{分数 >= 阈值?}:::decision
    ThresholdInfo[当前阈值: sim_threshold<br/>default: 0.55]:::config

    IsRecognized[识别结果<br/>is_recognized = True]:::output
    NotRecognized[识别结果<br/>is_recognized = False]:::output

    CalcDiag[计算诊断信息]:::process
    ThresholdDist[阈值距离<br/>distance = score - threshold]:::process
    CalcDiff[与第二接近分数的差异<br/>top2_diff]:::process

    QualityMetrics[质量指标]:::output

    BuildResult[构建返回结果]:::process
    AddBasic[添加基本信息<br/>is_recognized, confidence]:::process
    AddDiag[添加诊断信息<br/>threshold_distance, top2_diff]:::process
    AddPrep[添加预处理信息<br/>duration, sample_rate, rms_energy]:::process

    CheckResult{结果验证}:::decision

    Success[识别成功<br/>返回完整结果]:::output
    Fail[识别失败<br/>返回错误信息]:::error

    End([结束]):::output

    Start --> InputAudio
    Start --> InputPk

    Config -.->|参数| LoadModel
    Config -.->|参数| ApplyVAD
    Config -.->|参数| CheckCrop
    Config -.->|参数| BufferCheck
    Config -.->|参数| CompareThreshold
    Config -.->|参数| ScoreCompensation

    LoadModel --> LoadAudio
    LoadModel --> ExtractTest

    LoadAudio --> CheckFormat
    CheckFormat -->|WAV| Resample
    CheckFormat -->|M4A| ConvertM4a
    CheckFormat -->|其他| ConvertError
    ConvertM4a --> Resample

    Resample --> ToMono
    ToMono --> GetDuration
    GetDuration --> CheckDuration
    CheckDuration -->|正常| RecordInfo
    CheckDuration -->|空/过短| Fail
    RecordInfo --> ApplyVAD

    ApplyVAD -->|True| VadProcess
    ApplyVAD -->|False| SkipVAD

    VadProcess --> VadDone
    SkipVAD --> VadDone

    VadDone --> CheckCrop
    CheckCrop -->|full_utterance| CropFull
    CheckCrop -->|sliding_window| CropWindow

    CropFull --> BufferCheck
    CropWindow --> BufferCheck

    BufferCheck -->|> buffer_keep| Truncate
    BufferCheck -->|<= buffer_keep| NoTruncate

    Truncate --> CroppedAudio
    NoTruncate --> CroppedAudio

    CroppedAudio --> RecordCrop
    RecordCrop --> CalcRMS

    InputPk --> LoadRefPkl
    LoadRefPkl --> ValidatePkl
    ValidatePkl -->|有效| RefEmbedding
    ValidatePkl -->|无效| PklError

    RefEmbedding --> CheckRefDim
    CheckRefDim -->|256| ExtractTest
    CheckRefDim -->|其他| RefError

    CalcRMS --> RecordRMS
    RecordRMS --> ExtractTest

    ExtractTest --> Preprocess
    Preprocess --> ModelForward
    ModelForward --> TestNorm
    TestNorm --> TestEmbedding

    TestEmbedding --> ComputeSim
    RefEmbedding --> ComputeSim

    ComputeSim --> FinalScore

    FinalScore --> ScoreCompensation
    ScoreCompensation -->|True| GetVadDuration
    ScoreCompensation -->|False| NoCompensation

    GetVadDuration --> CalcCompensation
    CalcCompensation --> ApplyCompensation
    ApplyCompensation --> CompareThreshold
    NoCompensation --> CompareThreshold

    CompareThreshold -.->|参考| ThresholdInfo

    CompareThreshold -->|>= threshold| IsRecognized
    CompareThreshold -->|< threshold| NotRecognized

    IsRecognized --> CalcDiag
    NotRecognized --> CalcDiag

    CalcDiag --> ThresholdDist
    ThresholdDist --> CalcDiff
    CalcDiff --> QualityMetrics

    QualityMetrics --> BuildResult
    BuildResult --> AddBasic
    AddBasic --> AddDiag
    AddDiag --> AddPrep
    AddPrep --> CheckResult

    CheckResult -->|有效| Success
    CheckResult -->|无效| Fail

    Success --> End
    Fail --> End
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

### 2. VAD 处理（可选）

| 参数 | 值 | 说明 |
|------|------|------|
| threshold | 0.5 | VAD 置信度阈值 |
| min_speech_duration_ms | 250 | 最小语音时长 |
| min_silence_duration_ms | 100 | 最小静音时长 |
| window_size_samples | 512 | 窗口大小 |

**VAD 效果**：去除静音段，只保留有效语音片段

### 3. 音频裁剪策略

#### full_utterance（默认）
```
完整音频 → 检查时长
    ↓
超过 buffer_keep_secs?
    ↓ 是 → 保留最后 60 秒
    ↓ 否 → 保留完整音频
```

#### sliding_window
```
音频 → 滑动窗口
    ↓
窗口大小: 2-5 秒
    ↓
重叠: 50%
    ↓
取最高分窗口
```

### 4. Embedding 提取

```
处理后的音频 (16kHz, mono)
    ↓
归一化 [-1, 1]
    ↓
ResNet34 前向传播
    ↓
256 维向量
    ↓
L2 归一化: ||v|| = 1.0
```

### 5. 相似度计算

```python
# 余弦相似度（L2 归一化后等于点积）
score = dot(reference, test)
     = sum(ref[i] * test[i] for i in range(256))

# 范围: [-1, 1]
#   1.0  = 完全相同
#   0.0  = 正交
#  -1.0  = 完全相反
```

### 6. 分数补偿（可选）

当启用 `enable_score_compensation` 时：

```python
if vad_duration < target_duration:
    factor = target_duration / vad_duration
    adjusted_score = score * factor
```

**目的**：补偿短音频导致的分数偏低

### 7. 阈值判断

| 阈值 | 值 | 说明 |
|------|------|------|
| sim_threshold | 0.55 | 默认识别阈值 |
| 高安全 | 0.60-0.65 | 误接受率极低 |
| 高召回 | 0.45-0.50 | 误拒绝率极低 |

```
score >= threshold?
    ↓ 是 → is_recognized = True
    ↓ 否 → is_recognized = False
```

### 8. 诊断信息

| 字段 | 计算方式 | 说明 |
|------|----------|------|
| confidence | cosine_sim | 相似度分数 |
| threshold_distance | score - threshold | 与阈值的距离 |
| top2_diff | score1 - score2 | 前两名分数差 |
| duration | len(audio) / sr | 音频时长（秒） |
| sample_rate | 16000 | 采样率 |
| rms_energy | sqrt(mean(x²)) | RMS 能量 |

### 9. 返回结果

```python
{
    "is_recognized": bool,      # 是否识别成功
    "confidence": float,        # 相似度分数 [0, 1]
    "threshold": float,         # 使用的阈值
    "threshold_distance": float,  # 与阈值的距离
    "top2_diff": float,         # 前两名分数差
    "preprocessing": {
        "duration": float,      # 处理后时长
        "sample_rate": int,     # 采样率
        "rms_energy": float,    # RMS 能量
        "original_duration": float,  # 原始时长
        "vad_duration": float   # VAD 后时长
    }
}
```

## 识别流程参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sim_threshold` | 0.55 | 识别阈值 |
| `verify_crop_mode` | "full_utterance" | 裁剪模式 |
| `verify_buffer_keep_secs` | 60.0 | 最大保留时长 |
| `enable_vad` | False | 启用 VAD |
| `vad_threshold` | 0.5 | VAD 置信度 |
| `enable_score_compensation` | False | 启用分数补偿 |
| `score_compensation_target_duration` | 2.0 | 补偿目标时长 |

## 识别决策流程

```
                    ┌─────────────────┐
                    │  加载测试音频    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  预处理 & VAD   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  提取 Embedding │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼──────┐    ┌────────▼────────┐    ┌─────▼─────┐
│ 加载参考声纹  │    │  计算相似度     │    │ 质量评估   │
└───────┬──────┘    └────────┬────────┘    └─────┬─────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  分数补偿（可选）│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  与阈值比较      │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
         ┌──────▼──────┐          ┌──────▼──────┐
         │ score >= 0.55 │          │ score < 0.55 │
         └──────┬──────┘          └──────┬──────┘
                │                         │
         ┌──────▼──────┐          ┌──────▼──────┐
         │ 识别成功      │          │ 识别失败     │
         └─────────────┘          └─────────────┘
```

## 分数解释

| 分数范围 | 含义 | 说明 |
|---------|------|------|
| 0.70 - 1.00 | 非常匹配 | 极有可能是同一人 |
| 0.55 - 0.70 | 匹配 | 可能是同一人 |
| 0.45 - 0.55 | 模糊区 | 不确定，需要更多信息 |
| 0.30 - 0.45 | 不太匹配 | 可能不是同一人 |
| 0.00 - 0.30 | 不匹配 | 极可能不是同一人 |
| < 0.00 | 完全相反 | 不应该出现（检查错误） |
