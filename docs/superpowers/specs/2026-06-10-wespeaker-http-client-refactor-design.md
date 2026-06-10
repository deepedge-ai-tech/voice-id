# WeSpeaker HTTP Client Refactoring Design

## 概述

将 `WespeakerDeep` 从本地 ONNX 推理引擎重构为 voice-id REST API 的 HTTP 客户端，删除所有模型文件和 PyTorch 依赖，仅保留兼容的 API 接口。

## 背景

当前项目体积 ~230MB（含 PyTorch 模型 193MB、vendored wespeaker 924KB、ONNX 模型 29MB），依赖链复杂（torch、torchaudio、pyannote-audio 等）。现有语音识别服务 `voiceprint-api`（FastAPI, 端口 8005）已提供完整的注册/识别 HTTP API。

## 目标

1. 删除 `wespeaker_deep_dege.py`（PyTorch 引擎）和 `onnx_engine.py`（ONNX 引擎）
2. 清理所有不必要的代码：`client/`、`server/`、`_wespeaker/`、`_models/`、`_cohort/`、`diagnostics.py`、`reporters.py`、`realtime_monitor.py`、`asnorm.py`、`_utils.py`
3. 删除所有 `_voiceprints/*.pkl` 旧声纹文件
4. 保留 `_voiceprints/__init__.py`（索引→名称映射）
5. `WespeakerDeep` 改为 HTTP Client，**保持方法签名不变**
6. 重新注册 9 个内置声纹到服务端
7. 打包为 `.tar.gz`，其他项目 `pip install` 后可无缝使用

## 最终依赖

```toml
dependencies = [
    "requests>=2.28",
    "soundfile>=0.13",
]
```

删除了所有 ML 依赖（torch、torchaudio、onnxruntime、pyannote-audio 等）。

## 最终文件结构

```
src/wespeaker_deep_edge/
├── __init__.py              # 导出 WespeakerDeep, RecognitionResult, __version__
├── __main__.py              # CLI 入口（调 HTTP API）
├── client.py                # WespeakerDeep HTTP Client
└── _voiceprints/
    └── __init__.py          # 9 人索引↔名称映射
```

## WespeakerDeep API 设计

### 构造

```python
WespeakerDeep(
    base_url: str = "http://127.0.0.1:8005",    # 可从 env VOICE_ID_URL
    api_key: str = "",                           # 可从 env VOICE_ID_KEY
)
```

### enroll()

```python
def enroll(self, audio_path: str | Path, pk_path: str = "voice.pkl") -> dict
```

- `pk_path` 参数保留（兼容旧调用方），但不写本地 .pkl
- `speaker_id` 从 `pk_path` 文件名推断：`voice_john.pkl` → `john`，`voice.pkl` → 从音频文件名推断
- 内部调用 `POST /voiceprint/register`
- 返回 `{"ok": True, "msg": "已登记: john"}`

### recognize()

```python
def recognize(
    self,
    audio_path: str | Path | np.ndarray,
    voiceprint: np.ndarray | str | Path | None = None,
) -> dict
```

- `voiceprint` 为 `None` → 使用 load_templates 缓存的 ID 列表
- `voiceprint` 为路径 → 从文件名推断 speaker_id
- 内部调用 `POST /voiceprint/identify`
- 返回 `{"is_recognized": bool, "confidence": float, "name": str}`

### recognize_multi_pcm()

```python
def recognize_multi_pcm(
    self,
    pcm: np.ndarray,
    sample_rate: int = 16000,
) -> RecognitionResult
```

- 将 PCM 数组写为临时 WAV 文件（`soundfile.write`）
- 调用 `POST /voiceprint/identify`
- 删除临时文件
- 返回 `RecognitionResult(is_recognized, confidence, name)`

### load_templates()

```python
def load_templates(
    self,
    indices: list[int] | None = None,
    files: dict[str, str | Path] | None = None,
) -> None
```

- 仅缓存 `speaker_ids` 列表到 `self._speaker_ids`
- index → name 通过 `_voiceprints.get_voiceprint_name()` 解析
- files 的 key 直接作为 speaker_id

### 删除的方法

| 方法 | 原因 |
|------|------|
| `extract_embedding()` | 无 HTTP 端点对应 |
| `load_cohort()` | AS-Norm 是服务端职责 |
| `clear_templates()` | 可选保留，清空本地缓存 |
| `load()` | .pkl 文件不再本地管理 |

### 删除的类

`DeepConfig` — 不再需要，阈值由服务端控制。`is_recognized` 判断逻辑：

```python
# 服务端返回 {"speaker_id": "john", "score": 0.7031"}
# → speaker_id != "" 即为识别成功
is_recognized = bool(result["speaker_id"])
```

`RecognitionResult` 保留，但转换为 NamedTuple 的纯 Python 版本（无 numpy）。

## HTTP API 认证

服务端有两种认证方式，客户端需支持两种：

| 端点 | 认证方式 |
|------|---------|
| `GET /voiceprint/health?key=<token>` | URL 查询参数 |
| `POST /voiceprint/register` | Authorization: Bearer <token> |
| `POST /voiceprint/identify` | Authorization: Bearer <token> |
| `DELETE /voiceprint/{id}` | Authorization: Bearer <token> |

## 内置声纹索引映射

```python
_PEOPLE = [
    "john",             # 0  (John 和 John-double-mic 合并为同一人)
    "frank",            # 1
    "michael",          # 2
    "qingqing",         # 3
    "xixi",             # 4
    "zhong",            # 5
    "angle",            # 6
    "albert",           # 7
]
```

John-double-mic 与 John 是同一人在不同录音条件下的音频，合并为一个 `speaker_id`（john）。
注册时用 `John-double-mic.wav`（更长，13.7MB，包含更多语音特征变化）。

## 声纹重新注册

使用 `asset_combine/` 下的 WAV 文件注册所有 8 人：

| Index | speaker_id | 音频文件 |
|-------|-----------|---------|
| 0 | john | asset_combine/John-double-mic.wav |
| 1 | frank | asset_combine/Frank.wav |
| 2 | michael | asset_combine/Michael.wav |
| 3 | qingqing | asset_combine/Qingqing.wav |
| 4 | xixi | asset_combine/Xixi.wav |
| 5 | zhong | asset_combine/Zhong.wav |
| 6 | angle | asset_combine/angle.wav |
| 7 | albert | asset_combine/Albert.wav |

注册前先删除服务端上已有的旧声纹。

## 打包

```bash
tar -czf wespeaker-deep-edge-docker-v0.2.0.tar.gz \
    src/wespeaker_deep_edge/ \
    pyproject.toml \
    README.md
```

其他项目安装：

```bash
pip install wespeaker-deep-edge-docker-v0.2.0.tar.gz
```

## 删除的文件清单

| 文件/目录 | 大小 | 说明 |
|-----------|------|------|
| `src/wespeaker_deep_edge/wespeaker_deep_dege.py` | ~50KB | PyTorch 引擎 |
| `src/wespeaker_deep_edge/onnx_engine.py` | ~30KB | ONNX 引擎 |
| `src/wespeaker_deep_edge/asnorm.py` | ~15KB | AS-Norm |
| `src/wespeaker_deep_edge/_utils.py` | ~3KB | 工具函数（依赖 torch） |
| `src/wespeaker_deep_edge/diagnostics.py` | ~15KB | 诊断工具 |
| `src/wespeaker_deep_edge/reporters.py` | ~3KB | 报告工具 |
| `src/wespeaker_deep_edge/realtime_monitor.py` | ~10KB | 实时监控 |
| `src/wespeaker_deep_edge/client/` | ~10KB | WebSocket 客户端 |
| `src/wespeaker_deep_edge/server/` | ~20KB | WebSocket 服务端 |
| `src/wespeaker_deep_edge/_wespeaker/` | ~924KB | vendored wespeaker |
| `src/wespeaker_deep_edge/_models/vblinkf/avg_model.pt` | ~193MB | PyTorch 模型 |
| `src/wespeaker_deep_edge/_models/vblinkf/model.onnx` | ~29MB | ONNX 模型 |
| `src/wespeaker_deep_edge/_voiceprints/*.pkl` | ~44KB | 旧声纹文件 |
| `src/wespeaker_deep_edge/_cohort/` | ~320KB | Cohort 数据 |

## 向后兼容性

### 不变的

- `from wespeaker_deep_edge import WespeakerDeep` 继续可用
- `WespeakerDeep()` 构造方法签名兼容（新增 `base_url` / `api_key` 可选参数）
- `enroll(audio_path, pk_path)` 签名不变
- `recognize(audio_path, voiceprint)` 签名不变
- `recognize_multi_pcm(pcm, sample_rate)` 签名不变
- `load_templates(indices, files)` 签名不变
- `RecognitionResult` NamedTuple 结构不变

### 变化的（需通知调用方）

- `extract_embedding()` 删除，调用方需改用 `recognize()`
- `load()` 删除（不再本地管理 .pkl）
- 返回值 `confidence` 精度可能略有不同（服务端 vs 本地计算）
- `DeepConfig` 删除，调用方不再需要传配置对象（阈值由服务端控制）

## 测试策略

- `test_client.py`：Mock HTTP 响应，测试所有 WesepeakerDeep 方法
- 不需要模型文件，测试极轻量
- 覆盖率目标保持 ≥25%
