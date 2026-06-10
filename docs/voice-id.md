# 声纹识别 API 参考

## 服务信息

| 项目 | 说明 |
|------|------|
| 基础地址 | `http://<host>:8005` |
| 认证方式 | Bearer Token（URL 参数 `?key=` 或 Header `Authorization: Bearer <token>`） |
| 音频格式 | WAV（自动重采样到 16kHz，支持任意采样率输入） |
| 音频时长 | 0.5s ~ 30s，建议 ≥2s |

## 健康检查

```
GET /voiceprint/health?key=<token>
```

响应：

```json
{"total_voiceprints": 7, "status": "healthy"}
```

---

## 注册声纹

```
POST /voiceprint/register
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `speaker_id` | string | 说话人唯一标识 |
| `file` | file | WAV 音频文件 |

同一 `speaker_id` 重复注册会覆盖旧声纹。

响应：

```json
{"success": true, "msg": "已登记: zhangsan"}
```

curl 示例：

```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -F "speaker_id=zhangsan" \
  -F "file=@/path/to/audio.wav" \
  http://localhost:8005/voiceprint/register
```

---

## 声纹识别

```
POST /voiceprint/identify
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `speaker_ids` | string | 候选说话人 ID，逗号分隔（如 `zhangsan,lisi,wangwu`） |
| `file` | file | WAV 音频文件 |

返回最高相似度的说话人。

响应：

```json
{"speaker_id": "zhangsan", "score": 0.8521}
```

未匹配（所有候选相似度低于阈值，默认 0.2）：

```json
{"speaker_id": "", "score": 0.1635}
```

curl 示例：

```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -F "speaker_ids=zhangsan,lisi,wangwu" \
  -F "file=@/path/to/audio.wav" \
  http://localhost:8005/voiceprint/identify
```

---

## 删除声纹

```
DELETE /voiceprint/{speaker_id}
Authorization: Bearer <token>
```

响应：

```json
{"success": true, "msg": "已删除: zhangsan"}
```

---

## 性能参考

| 指标 | 数值（CPU） |
|------|------------|
| 识别延迟 | ~120ms |
| 注册延迟 | ~180ms |
| 模型推理 | ~114ms（占 95%） |
| 内存占用 | ~1.7 GB |
| 模型大小 | 29 MB |

---

## Python 调用示例

```python
import aiohttp

class VoiceprintClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def register(self, speaker_id: str, wav_bytes: bytes) -> bool:
        data = aiohttp.FormData()
        data.add_field("speaker_id", speaker_id)
        data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/voiceprint/register",
                                     headers=self.headers, data=data) as resp:
                result = await resp.json()
                return result.get("success", False)

    async def identify(self, speaker_ids: list[str], wav_bytes: bytes) -> tuple[str, float]:
        data = aiohttp.FormData()
        data.add_field("speaker_ids", ",".join(speaker_ids))
        data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/voiceprint/identify",
                                     headers=self.headers, data=data) as resp:
                result = await resp.json()
                return result.get("speaker_id", ""), result.get("score", 0)
```
