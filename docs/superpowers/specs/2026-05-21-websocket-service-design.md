# WebSocket 声纹识别服务设计

## 概述

为 Voice-ID (wespeaker-deep-edge) 项目添加 WebSocket 服务端和客户端 SDK，支持远程声纹注册和识别。同时支持 pip 可选安装 client/server 依赖。

## 架构

```
┌─────────────┐    WS (JSON header + binary)    ┌──────────────────┐
│  Client App │◄──────────────────────────────►│  WS Server       │
│  (调用方)   │                                 │  ─────────────    │
│             │  enroll(audio) → return id      │  EngineManager   │
│  .enroll()  │  load([id1, id2]) → ok          │  ├─ TemplateDB   │
│  .load()    │  recognize(audio) → {id, score} │  └─ WespeakerDeep│
│  .recognize │                                 └──────────────────┘
└─────────────┘
```

- 使用 `websockets` 库 + asyncio
- 服务端共享一个 `WespeakerDeep` 实例
- 每个客户端连接拥有独立的 `TemplateManager`

## 包结构

```
src/wespeaker_deep_edge/
├── __init__.py
├── __main__.py
├── _utils.py
├── wespeaker_deep_dege.py              # 核心引擎（不变）
├── _voiceprints/                        # 内置声纹（不变）
├── _models/                             # 内置模型（不变）
├── _wespeaker/                          # vendored wespeaker（不变）
│
├── server/                              # ★ 新增
│   ├── __init__.py
│   ├── cli.py                           #   服务端 CLI 入口
│   ├── ws_server.py                     #   WebSocket 协议处理
│   └── template_manager.py              #   多模板加载 + 矩阵批量比对
│
└── client/                              # ★ 新增
    ├── __init__.py                      #   导出 SpeakerClient
    └── speaker_client.py                #   WS 客户端 SDK
```

## WebSocket 协议

所有消息采用二进制帧模式：**第一条消息是 JSON 头部**（长度不固定，JSON 序列化），后续紧跟**音频二进制数据**（enroll/recognize 时），服务端回复单个 **JSON** 文本帧。

### enroll

```
C→S: {type: "enroll", id: "user_001"}
C→S: <binary audio bytes (raw wav)>
S→C: {status: "ok", data: {id: "user_001"}}
```

- `id`: 客户端指定的标识符，服务端作为 .pkl 文件名保存
- 服务端调用 `WespeakerDeep.enroll()` 提取 embedding 并保存到 `{storage_dir}/{id}.pkl`
- 返回注册后的 id

### load

```
C→S: {type: "load", ids: ["user_001", "preset_john"]}
S→C: {status: "ok", data: {loaded: 2, template_ids: ["user_001", "preset_john"]}}
```

- `ids` 可以是内置声纹名（`preset_john`, `preset_frank` 等前缀 `preset_`）或用户注册的 id
- 服务端从 `_voiceprints/` 或 `{storage_dir}/` 加载 .pkl 到内存 `TemplateManager`
- 返回实际加载成功的模板列表

### recognize

```
C→S: {type: "recognize"}
C→S: <binary audio bytes (raw wav)>
S→C: {status: "ok", data: {id: "preset_john", score: 0.8523}}
```

- 服务端提取音频 embedding，与当前 TemplateManager 中所有模板做矩阵批量 cosine similarity
- 返回最高分的 template_id 和分数

### error 响应

```
S→C: {status: "error", error: "音频中未检测到有效语音", code: "NO_SPEECH"}
```

错误码列表：
| code | 含义 |
|------|------|
| `FILE_NOT_FOUND` | 音频文件不存在 |
| `NO_SPEECH` | 音频无有效语音 |
| `TEMPLATE_NOT_FOUND` | 指定模板未加载 |
| `INVALID_PARAMS` | 请求参数错误 |
| `INTERNAL_ERROR` | 服务端内部错误 |

## 服务端设计

### WSServer

```python
class WSServer:
    """WebSocket 声纹识别服务。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 storage_dir: str = "./voiceprints",
                 model_path: str | None = None):
        ...

    async def start(self) -> None:
        """启动 WebSocket 服务，加载模型。"""

    async def stop(self) -> None:
        """优雅关闭。"""
```

- `storage_dir`: 用户注册声纹的存储目录，默认为 `./voiceprints`
- `model_path`: 模型路径，None 使用内置 vblinkf 模型

### TemplateManager

```python
class TemplateManager:
    """管理声纹模板，支持多模板矩阵批量比对。"""

    def __init__(self, wespeaker: WespeakerDeep,
                 voiceprints_dir: str, storage_dir: str):
        self._templates: dict[str, np.ndarray] = {}
        ...

    def load(self, ids: list[str]) -> list[str]:
        """加载多个 .pkl 到内存模板库。
        内置声纹用 ''preset_'' 前缀标识（preset_john, preset_frank...）。
        用户注册声纹从 storage_dir 加载。
        返回实际加载的 id 列表。
        """

    def recognize(self, audio_embedding: np.ndarray) -> tuple[str, float]:
        """矩阵计算批量余弦相似度：
        scores = (embeddings @ audio_embedding) / (||embeddings|| * ||audio_embedding||)
        返回 (best_id, max_score)
        """
        emb_matrix = np.stack(list(self._templates.values()))  # [N, 256]
        audio_vec = audio_embedding.reshape(1, -1)              # [1, 256]
        norm = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(audio_vec)
        scores = (emb_matrix @ audio_vec.T).flatten() / norm    # [N]
        best_idx = int(np.argmax(scores))
        return list(self._templates.keys())[best_idx], float(scores[best_idx])
```

### 协议处理流程

```python
async def _handle_connection(self, ws: WebSocketServerProtocol) -> None:
    mgr = TemplateManager(self._wespeaker, ...)  # 每个连接独立
    async for message in ws:
        if isinstance(message, str):  # 不应出现纯文本帧
            await ws.send(json.dumps({"status": "error", "error": "格式错误"}))
            return
        # 第一条消息是 JSON 头部
        header = json.loads(message)
        await self._handle_message(ws, mgr, header)
```

## 客户端设计

### SpeakerClient

```python
class SpeakerClientError(Exception):
    """客户端异常基类。"""

class SpeakerClient:
    """WebSocket 声纹客户端 SDK。封装连接细节，调用方无感。"""

    def __init__(self, url: str = "ws://localhost:8765",
                 auto_reconnect: bool = True):
        ...

    async def connect(self) -> None:
        """建立 WebSocket 连接。"""

    async def enroll(self, audio_path: str, id: str) -> str:
        """注册声纹。
        Args:
            audio_path: 本地音频文件路径
            id: 声纹标识符
        Returns:
            注册成功的 id
        Raises:
            SpeakerClientError: 注册失败
        """

    async def load(self, ids: list[str]) -> list[str]:
        """加载声纹模板到服务端内存。
        Args:
            ids: 声纹 ID 列表（支持预设声纹 preset_* 和用户注册 ID）
        Returns:
            实际加载的模板 ID 列表
        Raises:
            SpeakerClientError: 加载失败
        """

    async def recognize(self, audio_path: str) -> dict:
        """识别音频。
        Args:
            audio_path: 本地音频文件路径
        Returns:
            {"id": str, "score": float} 最高分模板 ID 和分数
        Raises:
            SpeakerClientError: 识别失败
        """

    async def close(self) -> None:
        """关闭连接。"""
```

- `auto_reconnect`: 连接断开时是否自动重连
- 内部错误统一转换为 `SpeakerClientError`，不暴露 WebSocket 异常
- `connect()` 一次，可重复调用 enroll/load/recognize

### 内部实现

```python
async def _send_request(self, msg_type: str,
                        audio_path: str | None = None,
                        **kwargs) -> dict:
    """发送请求并接收响应。"""
    header = {"type": msg_type, **kwargs}
    if audio_path:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        await self._ws.send(json.dumps(header).encode())
        await self._ws.send(audio_data)
    else:
        await self._ws.send(json.dumps(header).encode())

    resp = json.loads(await self._ws.recv())
    if resp["status"] == "error":
        raise SpeakerClientError(resp["error"])
    return resp["data"]
```

## CLI 入口

### 服务端启动

```bash
python -m wespeaker_deep_edge.server --host 0.0.0.0 --port 8765 --storage-dir ./voiceprints
```

`_server_cli.py`:
```python
import argparse
from .server.ws_server import WSServer

def main():
    parser = argparse.ArgumentParser(description="WeSpeaker WebSocket 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage-dir", default="./voiceprints")
    args = parser.parse_args()

    server = WSServer(host=args.host, port=args.port, storage_dir=args.storage_dir)
    asyncio.run(server.start())
```

## 依赖拆分 (pyproject.toml)

```toml
[project.optional-dependencies]
client = [
    "websockets>=12.0",
    "numpy==1.26.4",
    "soundfile>=0.13.1",
]
server = [
    "websockets>=12.0",
    "numpy==1.26.4",
    "soundfile>=0.13.1",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "pyannote-audio>=3.3.2",
    "scipy>=1.0",
    "pydub>=0.25.0",
    "pyyaml==6.0.3",
    "silero-vad",
    "audiomentations>=0.43.1",
    "tqdm",
    "hdbscan>=0.8.40",
    "kaldiio",
    "s3prl",
    "umap-learn==0.5.6",
    "accelerate",
    "onnxruntime>=1.16.0,<2.0",
    "openai-whisper",
    "peft",
    "scikit-learn",
    "seaborn>=0.13.2",
    "matplotlib>=3.10.9",
]
```

```toml
[project.scripts]
wespeaker-deep-edge = "wespeaker_deep_edge.client:main"
wespeaker-deep-edge-server = "wespeaker_deep_edge.server.cli:main"
```

```toml
[tool.setuptools.package-data]
"wespeaker_deep_edge" = ["*.yaml", "*.yml", "*.json", "*.txt"]
"wespeaker_deep_edge._models" = ["**/*"]
"wespeaker_deep_edge._voiceprints" = ["**/*"]
"wespeaker_deep_edge._wespeaker" = ["**/*"]
"wespeaker_deep_edge.server" = ["**/*"]
"wespeaker_deep_edge.client" = ["**/*"]
```

安装方式：
```bash
pip install wespeaker-deep-edge                # 完整安装（全部依赖）
pip install wespeaker-deep-edge[client]         # 仅客户端（轻量）
pip install wespeaker-deep-edge[server]         # 仅服务端
```

## 测试

- `tests/wespeaker_deep_edge/test_template_manager.py` — 矩阵计算正确性
- `tests/wespeaker_deep_edge/test_ws_server.py` — 协议处理（mock WS）
- `tests/wespeaker_deep_edge/test_speaker_client.py` — 客户端 SDK（mock WS）

## 预定义预设声纹 ID

| 内置声纹 | ID |
|----------|-----|
| John | `preset_john` |
| Frank | `preset_frank` |
| Michael | `preset_michael` |
| Qingqing | `preset_qingqing` |
| Xixi | `preset_xixi` |
| Zhong | `preset_zhong` |
| Angle | `preset_angle` |
| John_usb_yun | `preset_john_usb_yun` |
