# 系统总体架构图 (System Architecture Diagram)

```mermaid
graph TB
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef storage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef external fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef gateway fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef client fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef deep fill:#e0f2f1,stroke:#00796b,stroke-width:3px;

    subgraph 用户层
        CLI([CLI 命令行]):::client
        Script([Python 脚本]):::client
    end

    subgraph 核心层 — WespeakerClient (Legacy)
        Loader[音频加载 torchaudio/soundfile]:::core
        Preprocess[预处理 重采样/单声道]:::core
        VAD[VAD 静音检测 可选 默认关闭]:::core
        Client[WespeakerClient 主类]:::core
        Model[ResNet34 pyannote.audio 3.3+]:::core
        Embed[Embedding 提取 256维]:::core
    end

    subgraph 增强层 — WespeakerClient (Legacy)
        Augment[噪声增强 audiomentations]:::external
        SNR[SNR 估计]:::external
    end

    subgraph 核心层 — WespeakerDeep (当前最优)
        Deep[WespeakerDeep 主类]:::deep
        DeepConfig[DeepConfig 参数配置]:::deep
        MultiTemplate[多模板匹配 templates 列表]:::deep
        ScoreComp[分数补偿 sqrt 模式]:::deep
        HeadWindow[head_window 裁剪]:::deep
    end

    subgraph 工具模块
        Diag[diagnostics 诊断工具]:::gateway
        Report[reporters 报告生成]:::gateway
        Realtime[realtime_monitor 实时监控]:::gateway
    end

    subgraph 存储层
        Pkl[(声纹模板 .pkl dict格式)]:::storage
        ModelFile[(模型文件 _models/wespeaker/)]:::storage
        Asset[(音频素材 asset/)]:::storage
    end

    CLI --> Client
    CLI --> Deep
    Script --> Client
    Script --> Deep
    Client --> Loader
    Loader --> Preprocess
    Preprocess --> VAD
    VAD --> Client
    Client --> Model
    Model --> Embed
    Client --> Augment
    Client --> SNR
    Augment --> Client
    SNR --> Client
    Client --> Pkl
    Model --> ModelFile
    Loader --> Asset

    Deep --> DeepConfig
    Deep --> Client
    Deep --> MultiTemplate
    Deep --> ScoreComp
    Deep --> HeadWindow
    Deep --> Diag
    Deep --> Report
    Deep --> Pkl
    MultiTemplate --> Pkl
```
