# Voice-ID 项目结构与依赖

## 项目概述

WeSpeaker 声纹识别工具 — 独立的声纹注册与识别 CLI 工具。
基于 pyannote.audio 后端，支持从音频文件提取声纹 embedding 并保存为 `.pkl`，后续通过余弦相似度进行说话人验证。

- **包名**: `wespeaker-deep-edge` v0.1.23
- **Python**: 3.10.19
- **构建**: setuptools
- **包管理**: uv

---

## 目录结构

```
Voice-ID/
├── pyproject.toml                        # 项目配置与依赖声明
├── CLAUDE.md                             # 项目级 AI 指令
├── README.md                             # 项目说明
├── .gitignore                            # Git 忽略规则
│
├── src/wespeaker_deep_edge/              # ★ 主源码包
│   ├── __init__.py                       #   包入口，导出 DeepConfig, WespeakerDeep
│   ├── __main__.py                       #   CLI 入口 (python -m)
│   ├── _utils.py                         #   内部工具函数
│   ├── wespeaker_deep_dege.py            # ★ 核心：WespeakerDeep + DeepConfig
│   ├── diagnostics.py                    #   诊断分析工具
│   ├── reporters.py                      #   报告生成
│   ├── realtime_monitor.py               #   实时音频监控
│   │
│   ├── server/                           # ★ WebSocket 服务端
│   │   ├── __init__.py                   #   导出 WSServer, TemplateManager
│   │   ├── cli.py                        #   服务端 CLI 入口
│   │   ├── ws_server.py                  #   WebSocket 协议处理
│   │   └── template_manager.py           #   多模板加载 + 矩阵批量比对
│   │
│   ├── client/                           # ★ WebSocket 客户端 SDK
│   │   ├── __init__.py                   #   导出 SpeakerClient, SpeakerClientError
│   │   └── speaker_client.py             #   WS 客户端 SDK
│   │
│   ├── _voiceprints/                     #   内置声纹（打包进 whl）
│   │   ├── __init__.py                   #   get_voiceprint_path / get_voiceprint_name
│   │   ├── voice_john.pkl                #   John
│   │   ├── voice_frank.pkl               #   Frank
│   │   ├── voice_michael.pkl             #   Michael
│   │   ├── voice_qingqing.pkl            #   Qingqing
│   │   ├── voice_xixi.pkl                #   Xixi
│   │   ├── voice_zhong.pkl               #   Zhong
│   │   ├── voice_angle.pkl               #   Angle
│   │   └── voice_john_usb_yun.pkl        #   John (USB 云)
│   │
│   ├── _models/                          #   内置模型（打包进 whl）
│   │   ├── __init__.py
│   │   └── vblinkf/
│   │       ├── __init__.py
│   │       ├── avg_model.pt              #   VBLink-F 模型权重
│   │       └── config.yaml               #   模型配置
│   │
│   └── _wespeaker/                       # ★ vendored 官方 WeSpeaker 代码
│       └── wespeaker/
│           ├── __init__.py               #   导出 load_model, load_model_pt
│           ├── cli/                      #   CLI 入口
│           │   ├── hub.py
│           │   ├── speaker.py            #   load_model() 核心
│           │   └── utils.py
│           ├── models/                   #   模型架构定义
│           │   ├── resnet.py             #   ResNet
│           │   ├── ecapa_tdnn.py         #   ECAPA-TDNN
│           │   ├── campplus.py           #   CAMPPlus
│           │   ├── eres2net.py           #   ERes2Net
│           │   ├── res2net.py            #   Res2Net
│           │   ├── repvgg.py             #   RepVGG
│           │   ├── redimnet.py           #   ReDimNet
│           │   ├── samresnet.py          #   SAM-ResNet
│           │   ├── tdnn.py               #   TDNN
│           │   ├── xi_vector.py          #   xi-vector
│           │   ├── gemini_dfresnet.py    #   Gemini DFRN
│           │   ├── w2vbert_adapter_mfa.py #   W2VBERT Adapter MFA
│           │   ├── whisper_PMFA.py       #   Whisper PMFA
│           │   ├── speaker_model.py      #   SpeakerModel 基类
│           │   ├── pooling_layers.py     #   池化层
│           │   ├── projections.py        #   投影层
│           │   └── convert_repvgg.py     #   RepVGG 转换
│           ├── frontend/                 #   前端特征提取
│           │   ├── s3prl.py
│           │   ├── w2vbert.py
│           │   └── whisper_encoder.py
│           ├── dataset/                  #   数据集与数据加载
│           │   ├── dataset.py
│           │   ├── dataset_deprecated.py
│           │   ├── dataset_utils.py
│           │   ├── dataset_utils_deprecated.py
│           │   ├── lmdb_data.py
│           │   └── processor.py
│           ├── utils/                    #   工具函数
│           │   ├── utils.py
│           │   ├── checkpoint.py
│           │   ├── executor.py
│           │   ├── executor_deprecated.py
│           │   ├── file_utils.py
│           │   ├── schedulers.py
│           │   ├── score_metrics.py
│           │   ├── embedding_processing.py
│           │   └── plda/                 #   PLDA 工具
│           │       ├── kaldi_utils.py
│           │       ├── plda_utils.py
│           │       └── two_cov_plda.py
│           ├── diar/                     #   说话人日志
│           │   ├── extract_emb.py
│           │   ├── make_fbank.py
│           │   ├── make_oracle_sad.py
│           │   ├── make_rttm.py
│           │   ├── make_system_sad.py
│           │   ├── spectral_clusterer.py
│           │   └── umap_clusterer.py
│           ├── ssl/                      #   自监督学习
│           │   ├── bin/                  #   训练脚本
│           │   ├── dataset/
│           │   ├── models/               #   MoCo, SimCLR, DINO
│           │   └── utils/
│           └── bin/                      #   官方 CLI 工具
│               ├── extract.py
│               ├── score.py
│               ├── train.py
│               ├── export_onnx.py
│               ├── compute_det.py
│               ├── compute_metrics.py
│               ├── average_model.py
│               ├── adapt_plda.py
│               └── ...
│
├── tests/                               # ★ 测试套件
│   └── wespeaker_deep_edge/
│       ├── __init__.py
│       ├── test_wespeaker_deep_dege.py   #   WespeakerDeep 单元测试
│       ├── test_template_manager.py      #   TemplateManager 矩阵计算
│       ├── test_ws_server.py             #   WSServer 协议处理
│       └── test_speaker_client.py        #   SpeakerClient SDK 测试
│
├── scripts/                              # ★ 工具脚本
│   ├── best_recognition.py               #   最佳配置注册与识别
│   ├── cross_test.py                     #   交叉测试
│   ├── cross_test_merged.py              #   交叉测试（合并版）
│   ├── batch_cross_test.py               #   批量交叉测试
│   ├── batch_enroll_voiceprints.py       #   批量注册声纹
│   ├── split_registration.py             #   按静音切分注册音频
│   ├── test_sliding_window.py            #   滑动窗口测试
│   ├── sliding_window_analyzer.py        #   VAD 触发声纹识别分析
│   ├── convert_sample_rate.py            #   采样率转换
│   ├── generate_audio_variants.py        #   音频变体生成
│   ├── mix_audio.py                      #   音频混合
│   ├── mixed_voice_test.py               #   混合语音测试
│   ├── apply_aec_processing.py           #   AEC 处理
│   ├── backfill_confidence.py            #   置信度回填
│   ├── export_frank_vad.py               #   导出 Frank VAD
│   ├── export_vad_audios.py              #   导出 VAD 音频
│   ├── record_script.py                  #   录音脚本
│   ├── run_realtime_monitor.py           #   运行实时监控
│   ├── classify_orin.py                  #   Orin 分类
│   ├── official-demo.py                  #   官方 Demo
│   ├── official_cross_test.py            #   官方交叉测试
│   ├── plot_batch_summary.py             #   批量结果绘图
│   ├── plot_history.py                   #   历史趋势绘图
│   ├── test_script.py                    #   测试脚本
│   ├── test_whl_isolated.sh              #   whl 隔离环境测试
│   └── output/                           #   脚本输出
│       ├── cross_test_report_*.md
│       ├── cross_test_aggregate_*.png
│       └── official_cross_test_heatmap.png
│
├── docs/                                 # ★ 文档
│   ├── AS-Norm.md
│   ├── script.md
│   ├── test-plan.md
│   ├── optimization-status.md
│   ├── voice-id-experiments.html
│   ├── diagrams/                         #   Mermaid 架构图
│   │   ├── architecture.md
│   │   ├── data-flow.md
│   │   ├── modules.md
│   │   ├── recognition-flow.md
│   │   ├── registration-flow.md
│   │   ├── sequence.md
│   │   ├── tech-stack.md
│   │   ├── roadmap.md
│   │   └── test-configuration.md
│   │   ├── train-architecture.md
│   │   ├── train-data-flow.md
│   │   ├── train-modules.md
│   │   ├── train-sequence.md
│   │   ├── train-tech-stack.md
│   │   └── train-roadmap.md
│   └── superpowers/                      #   AI 实验计划
│       ├── plans/
│       └── specs/
│
├── asset/                                # 音频素材（Git 忽略）
├── models/                               # 模型符号链接（Git 忽略）
└── dist/                                 # whl 构建产物（Git 忽略）
```

---

## 模块依赖关系

### 内部依赖图

```
__main__.py (CLI 入口)
  ├── _voiceprints/__init__.py     (get_voiceprint_path, get_voiceprint_name)
  └── wespeaker_deep_dege.py       (DeepConfig, WespeakerDeep)

__init__.py (包入口)
  ├── diagnostics.py
  ├── realtime_monitor.py
  ├── reporters.py
  └── wespeaker_deep_dege.py       (DeepConfig, WespeakerDeep)

realtime_monitor.py
  └── wespeaker_deep_dege.py       (WespeakerDeep)

wespeaker_deep_dege.py ★ 核心模块
  ├── _wespeaker/wespeaker/         (vendored 官方 WeSpeaker)
  │   └── cli/speaker.py → load_model()
  ├── numpy
  ├── torch
  └── Python 标准库 (pickle, dataclasses, logging, sys)

diagnostics.py
  ├── numpy
  ├── torch (+ nn.functional)
  └── Python 标准库 (time, dataclasses)

_utils.py
  ├── torch
  └── Python 标准库 (functools, pathlib)

server/ws_server.py ★ WebSocket 服务
  ├── server/template_manager.py
  ├── client/wespeaker_deep_dege.py     (WespeakerDeep)
  ├── websockets
  ├── numpy
  └── Python 标准库 (json, tempfile, pathlib)

server/template_manager.py
  ├── client/wespeaker_deep_dege.py     (WespeakerDeep)
  ├── numpy (矩阵运算)
  └── Python 标准库 (pathlib)

client/speaker_client.py ★ 客户端 SDK
  ├── websockets
  └── Python 标准库 (json, pathlib)
```

### 外部依赖（pyproject.toml）

| 依赖 | 版本 | 用途 |
|------|------|------|
| **torch** | ==2.8.0 | 深度学习框架，模型推理 |
| **torchaudio** | ==2.8.0 | 音频 I/O 与处理 |
| **numpy** | ==1.26.4 | 数值计算，embedding 操作 |
| **pyannote-audio** | >=3.3.2 | 声纹 embedding 提取（ResNet34） |
| **scipy** | >=1.0 | 科学计算 |
| **sounddevice** | ==0.5.2 | 实时音频采集 |
| **soundfile** | >=0.13.1 | 音频文件读写 |
| **pydub** | >=0.25.0 | 音频处理（ffmpeg 封装） |
| **pyyaml** | ==6.0.3 | YAML 配置解析 |
| **silero-vad** | - | 语音活动检测（VAD） |
| **audiomentations** | >=0.43.1 | 音频数据增强（噪声注入） |
| **tqdm** | - | 进度条 |
| **hdbscan** | >=0.8.40 | 聚类分析 |
| **kaldiio** | - | Kaldi 格式 I/O |
| **s3prl** | - | 自监督语音表示 |
| **umap-learn** | ==0.5.6 | 降维可视化 |
| **accelerate** | - | PyTorch 加速 |
| **onnxruntime** | >=1.16.0,<2.0 | ONNX 模型推理 |
| **openai-whisper** | - | Whisper 模型支持 |
| **peft** | - | 参数高效微调 |
| **scikit-learn** | - | 机器学习工具 |
| **seaborn** | >=0.13.2 | 统计可视化 |
| **matplotlib** | >=3.10.9 | 绘图 |
| **websockets** | >=12.0 | WebSocket 通信（server/client） |

### 可选依赖（pip install wespeaker-deep-edge[client/server]）

| Extra | 依赖 | 说明 |
|-------|------|------|
| client | websockets, numpy, soundfile | 仅客户端 SDK，轻量（无 torch） |
| server | 全部依赖 | 完整服务端推理 |

### 开发依赖

| 依赖 | 用途 |
|------|------|
| pytest >=8.0 | 测试框架 |
| pytest-cov >=5.0 | 覆盖率报告 |
| black >=24.0 | 代码格式化 |
| isort >=5.0 | import 排序 |

---

## 核心 API

```
WespeakerDeep(config=DeepConfig)
  ├── enroll(audio_path, pk_path)     # 提取 embedding → 保存 .pkl
  ├── recognize(audio_path, pk_path=None)  # 比对 → {is_recognized, confidence}
  └── load(pk_path) → np.ndarray     # 加载 .pkl embedding
```

```
WSServer(host, port, storage_dir)     # WebSocket 服务端
  ├── start()                         # 启动 WS 服务
  └── stop()                          # 优雅停止

TemplateManager(wespeaker, ...)       # 多模板矩阵管理
  ├── load(ids) → list[str]           # 加载模板到内存
  └── recognize(embedding) → (id, score)  # 矩阵批量 cosine similarity

SpeakerClient(url)                    # WS 客户端 SDK
  ├── connect()                       # 建立连接
  ├── enroll(audio_path, id) → str    # 注册声纹
  ├── load(ids) → list[str]           # 加载模板
  ├── recognize(audio_path) → dict    # 识别 → {id, score}
  └── close()                         # 关闭连接
```

## 数据流

```
本地: audio.wav → load_model() → embedding → cosine_similarity → (通过/拒绝)

WebSocket:
  enroll:  客户端 → JSON头部(id) → 音频binary → 服务端 → 保存.pkl → 返回id
  load:    客户端 → JSON头部(ids) → 服务端 → 加载.pkl到内存矩阵 → 返回loaded
  recognize: 客户端 → JSON头部 → 音频binary → 服务端 → extract_embedding → 矩阵[Id,John]×音频 → max(id,score)
```

## WebSocket 协议

二进制帧模式：**JSON 头部帧 + 音频数据帧**（enroll/recognize），服务端回复 JSON。

| 操作 | 请求 | 响应 |
|------|------|------|
| enroll | `{type:"enroll", id:"user_001"}` + binary | `{status:"ok", data:{id:"user_001"}}` |
| load | `{type:"load", ids:["preset_john","user_001"]}` | `{status:"ok", data:{loaded:2, template_ids:[]}}` |
| recognize | `{type:"recognize"}` + binary | `{status:"ok", data:{id:"preset_john", score:0.8523}}` |
| error | - | `{status:"error", error:"...", code:"NO_SPEECH"}` |

模板矩阵计算：load 时多个 embedding 堆叠为 `[N, 256]` 矩阵，recognize 时一次 `np.matmul` 批量算分。`preset_` 前缀标识内置声纹。

## 安装方式

```bash
pip install wespeaker-deep-edge                # 完整安装（全部依赖）
pip install wespeaker-deep-edge[client]         # 仅客户端（轻量，无 torch）
pip install wespeaker-deep-edge[server]         # 仅服务端
```

---

*生成时间: 2026-05-21*
