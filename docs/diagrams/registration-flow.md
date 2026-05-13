# 注册流程详细图 (Detailed Registration Flow)

## 完整注册流程

```mermaid
flowchart TD
    classDef input fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef config fill:#e0e0e0,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px;

    Start([开始注册]):::input
    InputPath[输入音频路径]:::input
    InputPk[输出 .pkl 路径]:::input

    Config[配置参数:<br/>sample_rate: 16000<br/>enable_vad: False<br/>noise_augment: True<br/>snr_levels: [20,15,10,5,0]]:::config

    LoadModel[加载 ResNet34 模型<br/>pyannote.audio 3.1]:::process

    LoadAudio[加载音频文件<br/>torchaudio.load]:::process
    CheckFormat{文件格式?}:::decision
    ConvertM4a[转换 M4A → WAV<br/>ffmpeg]:::process
    ConvertError[错误: 不支持的格式]:::error

    Resample[重采样到 16kHz<br/>torchaudio.transforms.Resample]:::process
    ToMono[转为单声道<br/>waveform.mean]:::process

    SplitAudio[切分音频片段<br/>按静音间隔]:::process
    VadSplit[Silero VAD 检测<br/>threshold: 0.5<br/>min_speech: 250ms<br/>min_silence: 100ms]:::process

    CheckDuration{片段时长 >= 1s?}:::decision
    FilterShort[过滤短片段]:::process

    Fragments[有效片段列表<br/>N fragments]:::output

    LoadNoise[加载噪声音频<br/>用于噪声增强]:::process
    ExtractNoise[提取噪声 profile<br/>随机采样 5 秒]:::process

    LoopStart[开始片段循环]:::process

    LoopEach{遍历每个片段}:::decision

    AugStart[噪声增强阶段]:::process

    LoopSNR[遍历 SNR 级别<br/>[20, 15, 10, 5, 0] dB]:::process

    MixAudio[混合噪声<br/>noise * 10^(-snr/20)]:::process
    AddNoise[original + scaled_noise]:::process

    AugDone[增强完成<br/>N × 5 个增强片段]:::output

    ExtractEmb[提取 Embedding]:::process
    Preprocess[预处理<br/>归一化 [-1, 1]]:::process
    ModelForward[模型前向传播<br/>ResNet34 → 256 维向量]:::process
    L2Norm[L2 归一化<br/>||v|| = 1.0]:::process

    CollectEmbed[收集所有 embeddings]:::process
    EmbList[Embedding 列表<br/>N × 5 个 256 维向量]:::output

    AvgStrategy{平均策略}:::decision

    SimpleAvg[简单平均<br/>sum / N]:::process
    SNRAvg[SNR 加权平均<br/>高 SNR 更高权重]:::process

    FinalNorm[最终 L2 归一化]:::process
    FinalEmb[最终声纹 Embedding<br/>256 维向量]:::output

    CheckNaN{检查 NaN/Inf?}:::decision
    HandleNaN[处理 NaN<br/>替换为零向量]:::error

    QualityCheck[质量评估]:::process
    CalcCompactness[计算紧密度<br/>within_class_compactness]:::process
    CalcDistance[计算余弦距离<br/>cosine_distances]:::process
    QualityReport[生成质量报告]:::output

    SavePkl[保存到 .pkl 文件<br/>pickle.dump]:::process
    WriteMeta[写入元数据<br/>num_segments, snr_levels]:::process

    CheckQuality{质量检查通过?}:::decision

    Success[注册成功<br/>返回: {ok, num_segments, pk_path}]:::output
    Fail[注册失败<br/>返回: {ok, error}]:::error

    End([结束]):::output

    Start --> InputPath
    Start --> InputPk
    InputPath --> LoadAudio
    InputPk --> SavePkl

    Config -.->|参数| LoadModel
    Config -.->|参数| VadSplit
    Config -.->|参数| LoopSNR
    Config -.->|参数| AvgStrategy

    LoadModel --> LoadAudio

    LoadAudio --> CheckFormat
    CheckFormat -->|WAV| Resample
    CheckFormat -->|M4A| ConvertM4a
    CheckFormat -->|其他| ConvertError
    ConvertM4a --> Resample

    Resample --> ToMono
    ToMono --> SplitAudio
    SplitAudio --> VadSplit
    VadSplit --> CheckDuration
    CheckDuration -->|>= 1s| Fragments
    CheckDuration -->|< 1s| FilterShort
    FilterShort --> LoopEach

    Fragments --> LoadNoise
    LoadNoise --> ExtractNoise
    ExtractNoise --> LoopStart

    LoopStart --> LoopEach

    LoopEach -->|还有片段| AugStart
    LoopEach -->|无片段| Fail

    AugStart --> LoopSNR
    LoopSNR --> MixAudio
    MixAudio --> AddNoise
    AddNoise --> LoopSNR
    LoopSNR -->|SNR 完成| ExtractEmb

    ExtractEmb --> Preprocess
    Preprocess --> ModelForward
    ModelForward --> L2Norm
    L2Norm --> CollectEmbed

    CollectEmbed --> LoopEach
    LoopEach -->|收集完成| AvgStrategy

    AvgStrategy -->|简单| SimpleAvg
    AvgStrategy -->|SNR加权| SNRAvg

    SimpleAvg --> FinalNorm
    SNRAvg --> FinalNorm

    FinalNorm --> FinalEmb
    FinalEmb --> CheckNaN

    CheckNaN -->|有 NaN| HandleNaN
    CheckNaN -->|正常| QualityCheck
    HandleNaN --> QualityCheck

    QualityCheck --> CalcCompactness
    CalcCompactness --> CalcDistance
    CalcDistance --> QualityReport
    QualityReport --> CheckQuality

    CheckQuality -->|通过| SavePkl
    CheckQuality -->|失败| Fail

    SavePkl --> WriteMeta
    WriteMeta --> Success

    Success --> End
    Fail --> End
```

## 注册流程关键步骤

### 1. 音频加载与预处理

```
原始音频 (.wav/.m4a)
    ↓
格式转换 (M4A → WAV，如需要)
    ↓
重采样 → 16kHz
    ↓
单声道转换 (stereo → mono)
    ↓
按静音切分片段
```

### 2. 片段过滤

| 条件 | 阈值 | 说明 |
|------|------|------|
| 最小时长 | ≥ 1.0 秒 | 过滤太短的片段 |
| VAD 阈值 | 0.5 | Silero VAD 置信度 |
| 最小语音 | 250 ms | 单次语音最小时长 |
| 最小静音 | 100 ms | 语音间隔判定 |

### 3. 噪声增强

对每个有效片段，应用 5 个 SNR 级别的噪声注入：

```python
for snr in [20, 15, 10, 5, 0]:  # dB
    # 计算噪声缩放因子
    scale = 10 ** (-snr / 20)
    # 混合
    augmented = original + noise * scale
```

**效果**：
- 20 dB: 轻微噪声
- 10 dB: 中等噪声
- 0 dB: 强噪声

### 4. Embedding 提取

```
音频片段 (16kHz, mono)
    ↓
预处理: 归一化到 [-1, 1]
    ↓
ResNet34 模型前向传播
    ↓
256 维向量
    ↓
L2 归一化: ||v|| = 1.0
```

### 5. 平均策略

#### 简单平均
```python
final = sum(embeddings) / len(embeddings)
final = final / ||final||  # L2 归一化
```

#### SNR 加权平均
```python
weights = [10 ** (snr / 20) for snr in snr_levels]
final = sum(e * w for e, w in zip(embeddings, weights))
final = final / ||final||
```

### 6. 质量评估

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| within_class_compactness | mean(cosine_distances) | 同类紧密度 |
| fragment_variance | std(embeddings) | 片段方差 |
| num_segments | len(fragments) × 5 | 有效片段数 |

### 7. 输出格式

```python
# .pkl 文件内容
{
    "embedding": np.ndarray  # 256 维向量
    "metadata": {
        "num_fragments": int,
        "snr_levels": list[int],
        "timestamp": str,
        "quality_metrics": dict
    }
}
```

## 注册流程参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_rate` | 16000 | 目标采样率 |
| `vad_threshold` | 0.5 | VAD 置信度阈值 |
| `min_segment_duration` | 1.0 | 最小片段时长（秒） |
| `enable_noise_augment` | True | 启用噪声增强 |
| `snr_levels` | [20,15,10,5,0] | SNR 级别列表 |
| `avg_strategy` | "simple" | 平均策略 |
