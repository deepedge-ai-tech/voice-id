# 数据流图 (Data Flow Diagram)

## 最佳配置参数 (WespeakerDeep / DeepConfig)

| 参数 | 值 | 说明 |
|------|------|------|
| **sim_threshold** | **0.50** | 余弦相似度阈值，18 轮实验验证 |
| **verify_crop_mode** | **head_window** | 超长音频保留头部 |
| **verify_buffer_keep_secs** | 60.0 | 最大保留时长 |
| **enable_vad** | False | 完整音频得分更高（默认关闭 VAD） |
| **enable_score_compensation** | **True** | sqrt 模式补偿短音频 |
| **score_compensation_mode** | sqrt | factor = min((target/dur)^0.5, 2.0) |
| **enroll_skip_vad** | True | 注册跳过 VAD |
| **enroll_clean_only** | True | 纯干净注册，不注入噪声 |
| **enable_multi_template** | True | 多模板匹配（取 max） |
| **sample_rate** | 16000 Hz | 统一采样率 |
| **short_audio_max_duration** | 1.5s | 短音频判定阈值 |

## 数据流图

```mermaid
flowchart LR
    classDef source fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef store fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef external fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef config fill:#e0e0e0,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;

    Config[("DeepConfig 参数:<br/>sim_threshold: 0.50<br/>crop_mode: head_window<br/>enable_vad: False<br/>score_comp: sqrt<br/>multi_template: True<br/>clean_only: True<br/>sample_rate: 16kHz")]:::config

    subgraph 注册流程 — 纯净注册
        RegAudio([注册音频目录 .wav]):::source
        RegLoad["音频加载<br/>sample_rate: 16kHz<br/>mono: True"]:::process
        RegLoop["遍历每文件<br/>独立处理"]:::process
        RegSkipVAD["跳过 VAD<br/>enroll_skip_vad=True"]:::process
        RegEmb["提取 Embedding<br/>模型: pyannote.audio<br/>维度: 256"]:::process
        RegNorm["L2 归一化<br/>||v|| = 1.0"]:::process
        RegCollect["收集所有 embeddings<br/>templates 列表"]:::process
        RegRef["计算 reference<br/>templates 均值 + 归一化"]:::process
        RegSave[("保存 .pkl  dict格式<br/>version: 1<br/>templates: [...]<br/>reference: [...]")]:::store

        RegAudio --> RegLoad
        RegLoad --> RegLoop
        RegLoop --> RegSkipVAD
        RegSkipVAD --> RegEmb
        RegEmb --> RegNorm
        RegNorm --> RegCollect
        RegCollect --> RegRef
        RegRef --> RegSave
    end

    subgraph 识别流程 — 多模板 + 分数补偿
        RecAudio([测试音频]):::source
        RecLoad["音频加载<br/>sample_rate: 16kHz"]:::process
        RecCrop["head_window 裁剪<br/>保留头部 N 秒<br/>buffer_keep: 60s"]:::process
        RecVAD["VAD 处理<br/>默认关闭"]:::process
        RecEmb["提取 Embedding<br/>模型: pyannote.audio"]:::process
        RecLoadPkl[("加载 .pkl<br/>所有 template 向量")]:::store
        RecMulti["多模板匹配<br/>max(cosine(emb, t) for t in templates)"]:::process
        RecComp["sqrt 分数补偿<br/>factor = (target/dur)^0.5<br/>comp = min(score * factor, 1.0)"]:::process
        RecDec{"补偿后分数 >= 0.50?"}:::output
        RecPass([识别成功]):::output
        RecFail([识别失败]):::output
        RecDiag["诊断信息<br/>num_templates_used<br/>sliding_windows_used<br/>compensation_factor"]:::process

        RecAudio -->|音频数据| RecLoad
        RecLoad -->|waveform| RecCrop
        RecCrop -->|处理后音频| RecVAD
        RecVAD -->|VAD处理后| RecEmb
        RecEmb -->|test_embedding| RecMulti
        RecLoadPkl -->|templates 列表| RecMulti
        RecMulti -->|raw_score| RecComp
        RecComp -->|compensated_score| RecDec
        RecMulti -->|所有分数| RecDiag
        RecDec -->|是| RecPass
        RecDec -->|否| RecFail
    end

    Config -.->|参数配置| RegSkipVAD
    Config -.->|参数配置| RegEmb
    Config -.->|参数配置| RecCrop
    Config -.->|参数配置| RecVAD
    Config -.->|参数配置| RecMulti
    Config -.->|参数配置| RecComp
    Config -.->|参数配置| RecDec

    RegSave -.->|模板列表| RecLoadPkl
```
