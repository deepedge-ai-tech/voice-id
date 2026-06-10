# WeSpeaker — Voice-ID HTTP Client

speaker-deep-edge 是一个轻量级 HTTP 客户端库，通过 REST API 调用远程声纹识别服务（voice-id / voiceprint-api）。

**没有 PyTorch 依赖，没有 ONNX Runtime 依赖** — 只需 `requests` + `soundfile`，安装即用。

## 特性

- 纯 HTTP 客户端 — 无本地模型，无 GPU，无深度学习框架
- 支持声纹注册（enroll）和识别（recognize）
- 支持 PCM 数组、本地 WAV 文件输入
- 内置 8 人声纹名称映射（索引 0-7），无需外部文件即可识别
- CLI 工具 + Python API，开箱即用
- 自动重采样音视频格式（服务端支持）

## 安装

### 从 PyPI（已发布）

```bash
pip install wespeaker-deep-edge
```

### 从 GitHub 直接安装

```bash
pip install git+https://github.com/deepedge-ai-tech/voice-id.git
```

### 从 tar 包安装（Docker 部署）

```bash
# 在项目 release 页面下载 wespeaker-deep-edge-docker-*.tar.gz
pip install wespeaker-deep-edge-docker-v0.2.0.tar.gz
```

### 从源码安装

```bash
git clone https://github.com/deepedge-ai-tech/voice-id.git
cd voice-id
pip install .
```

### 查看版本

```bash
python -m wespeaker_deep_edge --version
```

## 快速开始

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VOICE_ID_URL` | `http://192.168.5.9:8005` | 声纹识别服务地址 |
| `VOICE_ID_KEY` | `""` | API 认证密钥 |

### 命令行使用

```bash
# 注册声纹（会向 API 发送注册请求）
wespeaker-deep-edge enroll john audio.wav

# 识别声纹（使用内置 John 声纹，无需指定 speaker_id）
wespeaker-deep-edge recognize test.wav

# 识别声纹（指定候选说话人）
wespeaker-deep-edge recognize test.wav john,frank,albert

# 列出所有内置声纹
wespeaker-deep-edge list-voiceprints

# 指定 API 地址和密钥
wespeaker-deep-edge --url http://voice-id.example.com:8005 --key my-secret-token recognize test.wav
```

### 内置声纹索引

| Index | Name              |
|-------|-------------------|
| 0     | John              |
| 1     | Frank             |
| 2     | Michael           |
| 3     | Qingqing          |
| 4     | Xixi              |
| 5     | Zhong             |
| 6     | Angle             |
| 7     | Albert            |

CLI 默认使用 John (index 0)。索引仅用于 `load_templates(indices=[...])` 方法，实际识别时发送 `speaker_id` 到 API。

### Python API 使用

```python
from wespeaker_deep_edge import WespeakerDeep

# 创建客户端（默认使用环境变量或 localhost）
client = WespeakerDeep(
    base_url="http://voice-id.example.com:8005",
    api_key="your-api-key",
)

# 注册声纹
result = client.enroll("audio.wav", "voice_john.pkl")
print(result)  # {"ok": True, "msg": "已登记: john"}

# 加载候选声纹（内置索引）
client.load_templates(indices=[0])  # 使用 John

# 识别声纹（文件路径）
result = client.recognize("test.wav")
print(f"识别结果: {result}")
# {"is_recognized": True, "confidence": 0.85, "name": "john"}

# 加载多个候选
client.load_templates(indices=[0, 1, 2])  # John, Frank, Michael
result = client.recognize("test.wav")

# 使用自定义 speaker IDs
client.load_templates(files={"john": "voice_john.pkl", "frank": "voice_frank.pkl"})
result = client.recognize("test.wav")
```

### PCM 实时识别

```python
import soundfile as sf
from wespeaker_deep_edge import WespeakerDeep, RecognitionResult

client = WespeakerDeep()
client.load_templates(indices=[0])

# 读取 PCM 数据
pcm, sr = sf.read("test.wav")

# 识别
result: RecognitionResult = client.recognize_multi_pcm(pcm, sample_rate=sr)
if result.is_recognized:
    print(f"说话人: {result.name}, 置信度: {result.confidence:.4f}")
```

## Docker 部署

客户端与声纹识别服务分开部署。服务端提供了 Docker 镜像，你可以在项目 release 页面找到构建好的 tar 包：

```
wespeaker-deep-edge-docker-v0.2.0.tar.gz
```

部署后，客户端通过以下方式连接：

```python
import os
from wespeaker_deep_edge import WespeakerDeep

client = WespeakerDeep(
    base_url=os.getenv("VOICE_ID_URL", "http://192.168.5.9:8005"),
    api_key=os.getenv("VOICE_ID_KEY", ""),
)
```

## 项目结构

```
Voice-ID/
├── pyproject.toml                    # 项目配置与依赖
├── src/wespeaker_deep_edge/          # ★ 主源码包
│   ├── client.py                     #   HTTP 客户端 (WespeakerDeep)
│   ├── __init__.py                   #   包入口
│   ├── __main__.py                   #   CLI 入口 (python -m)
│   └── _voiceprints/                 #   内置声纹名称映射（8人）
├── tests/                            # 测试套件
└── docs/                             # 文档
    └── diagrams/architecture.md      # 架构图
```

## 架构

请参考 [docs/diagrams/architecture.md](docs/diagrams/architecture.md) 了解系统整体架构。

客户端架构非常简单：

1. `WespeakerDeep` 将音频文件/ PCM 发送到 `voice-id` 服务的 REST API
2. 服务端执行声纹注册或识别
3. 客户端解析 JSON 响应并返回结构化的 `RecognitionResult`

## 依赖

### 核心依赖

- Python 3.10+
- requests >= 2.28
- soundfile >= 0.13

### 开发依赖

- pytest >= 8.0
- pytest-cov >= 5.0
- pytest-asyncio >= 0.21
- black >= 24.0
- isort >= 5.0

## REST API 参考

详细的 API 文档请参考 [docs/voice-id.md](docs/voice-id.md)。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/voiceprint/register` | POST | 注册声纹（multipart: speaker_id + file） |
| `/voiceprint/identify` | POST | 声纹识别（multipart: speaker_ids + file） |
| `/voiceprint/health` | GET | 健康检查 |
| `/voiceprint/{speaker_id}` | DELETE | 删除声纹 |

## 开发

```bash
# 安装开发依赖
uv sync --group dev

# 运行测试
uv run pytest

# 代码格式化
uv run black . && uv run isort .
```

## 许可证

MIT License
