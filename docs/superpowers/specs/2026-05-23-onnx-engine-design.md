# OnnxEngine 轻量化声纹识别引擎

## 背景

当前 `WespeakerDeep` 依赖 PyTorch + torchaudio，`import torch` 就要消耗 300-500 MB 内存，
而模型本身（ResNet34, 256-d embedding）仅 25 MB。对于 Jetson Orin Nano 等边缘设备，PyTorch
的安装体积和运行时内存开销都过大。

## 目标

- 去掉 `torch` 和 `torchaudio` 依赖
- 只保留 `load_templates()` + `recognize_multi_pcm()` 两个核心功能
- 用 ONNX Runtime 替代 PyTorch 进行模型推理
- 用纯 numpy + scipy 替代 torchaudio 进行 FBANK 特征提取
- 运行时内存从 ~300-500 MB 降到 ~30-50 MB
- Mac CPU / Jetson Orin Nano (CUDA+TensorRT) 使用同一个 `.onnx` 模型文件

## 架构

```
PCM int16 ──→ [numpy fbank] ──→ [ONNX Runtime] ──→ [template match] ──→ [AS-Norm] ──→ RecognitionResult
                 scipy              model.onnx          N×256 matrix       300×300 cohort
```

各组件职责：

| 组件 | 职责 | 替代目标 |
|------|------|----------|
| `compute_fbank()` | PCM → 80-dim log-Mel FBANK + CMN | `torchaudio.compliance.kaldi.fbank` |
| `ONNX Runtime` | FBANK → 256-d embedding | `model.extract_embedding()` (PyTorch) |
| `template_match` | 余弦相似度矩阵乘法找最佳匹配 | 不变 |
| `CohortCache` | AS-Norm 分数归一化 | 不变 |

## API 设计

```python
class OnnxEngine:
    def __init__(
        self,
        model_path: str | Path | None = None,   # None → 内置 model.onnx
        providers: list[str] | None = None,      # None → ["CPUExecutionProvider"]
    )

    def load_templates(
        self,
        indices: list[int] | None = None,        # 内置声纹索引
        files: dict[str, str | Path] | None = None,  # 外部 .pkl
    ) -> None

    def clear_templates(self) -> None

    def recognize_multi_pcm(
        self,
        pcm: np.ndarray,                         # int16 或 float32
        sample_rate: int = 16000,
    ) -> RecognitionResult

    @property
    def config(self) -> OnnxConfig
```

`RecognitionResult` 保持现有兼容：

```python
class RecognitionResult(NamedTuple):
    is_recognized: bool
    confidence: float
    name: str
    all_scores: dict | None
```

`OnnxConfig` 精简配置：

```python
@dataclass
class OnnxConfig:
    sim_threshold: float = 0.55
    enable_asnorm: bool = True
    asnorm_threshold: float = 6.0
    asnorm_top_k: int = 300
    asnorm_norm_type: str = "asnorm"
```

## 文件变动

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/wespeaker_deep_edge/onnx_engine.py` | 新增 | OnnxEngine 主类 |
| `src/wespeaker_deep_edge/__init__.py` | 修改 | 导出 OnnxEngine |
| `scripts/export_onnx_model.py` | 新增 | 一次性 ONNX 导出（有 torch 环境跑） |
| `src/wespeaker_deep_edge/_models/vblinkf/model.onnx` | 产物 | 导出的 ONNX 模型（提交 git） |
| `pyproject.toml` | 修改 | 依赖变更 |
| `tests/` | 新增/修改 | onnx_engine 测试 |

## 依赖变更

```toml
[project]
dependencies = [
    "numpy",
    "scipy",        # FBANK + resample
    "soundfile",    # 音频文件读写
]

[project.optional-dependencies]
cpu = ["onnxruntime"]           # Mac, Linux x86
gpu = ["onnxruntime-gpu"]       # Jetson Orin (JetPack)
```

默认安装只有 numpy + scipy，运行时检测不到 onnxruntime 时提示用户按平台安装对应 extras。

## FBANK 实现

纯 numpy + scipy，参数对齐训练配置：

| 参数 | 值 | 来源 |
|------|-----|------|
| sample_rate | 16000 | config.yaml |
| num_mel_bins | 80 | config.yaml |
| frame_length | 25ms (400 samples) | config.yaml |
| frame_shift | 10ms (160 samples) | config.yaml |
| window_type | hamming | speaker.py |
| fft_size | 512 | 标准值 |
| low_freq | 20 Hz | Kaldi 默认 |
| high_freq | 7600 Hz | nyquist - 400 |
| CMN | 时间维度减均值 | speaker.py |

实现要点：
- scipy.signal.resample 用于重采样（代替 torchaudio.transforms.Resample）
- numpy.fft.rfft 计算功率谱
- 三角形 Mel 滤波器组，HTK mel 公式
- 自然对数 (np.log)，与 Kaldi 一致

## AS-Norm

直接复用现有的 `asnorm.py`（`CohortCache`），不需要任何修改。`asnorm.py` 只有 numpy 依赖。

## 错误处理

| 场景 | 行为 |
|------|------|
| 音频太短 (< 25ms) | `ValueError("audio too short")` |
| ONNX 模型不存在 | `FileNotFoundError` + 提示先跑 export 脚本 |
| 模板未加载 | `RuntimeError("call load_templates() first")` |
| cohort 文件不存在 | `logger.warning`，静默跳过 AS-Norm |
| onnxruntime 未安装 | 导入时 `ImportError` + 提示安装指令 |

## 测试策略

新建 `tests/test_onnx_engine.py`，覆盖：

- `compute_fbank` 输出形状和数值范围正确
- `extract_embedding` 输出维度 (256,) float32
- `recognize_multi_pcm` 与现有 `WespeakerDeep` 结果 cosine sim 偏差 < 0.01
- AS-Norm 打开/关闭分支
- 短音频边界情况
- 空模板报错

## 平台部署

**Mac（开发）**：
```
pip install wespeaker-deep-edge[cpu]
```

**Jetson Orin Nano（生产）**：
```
# JetPack 环境自带 onnxruntime-gpu
pip install wespeaker-deep-edge[gpu]
```

Provider 配置：
```python
# Mac
OnnxEngine()  # 默认 CPUExecutionProvider

# Orin Nano
OnnxEngine(providers=[
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
])
```

## 不做的功能

- 不保留 `enroll()` API（用外部脚本注册，ONNX 引擎只做识别）
- 不保留 `realtime_monitor` 对 PyTorch 的依赖
- 不保留 `diagnostics` 和 `reporters`（它们是开发辅助工具）
