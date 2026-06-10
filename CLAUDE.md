# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WeSpeaker Voice-ID HTTP Client — a lightweight `requests` + `soundfile` library wrapping the voice-id REST API for speaker enrollment and recognition.

**No PyTorch, no ONNX Runtime, no local model.** All recognition is delegated to a remote voiceprint-api service (`voiceprint-api` project on port 8005).

## Architecture

```
Voice-ID/
├── pyproject.toml                    # Dependencies: requests, soundfile
├── src/wespeaker_deep_edge/          # ★ Main package
│   ├── client.py                     #   WespeakerDeep class (HTTP client)
│   ├── __init__.py                   #   Exports WespeakerDeep, RecognitionResult
│   ├── __main__.py                   #   CLI (python -m wespeaker_deep_edge)
│   └── _voiceprints/                 #   Speaker name index (8 people)
├── tests/
│   └── wespeaker_deep_edge/
│       └── test_client.py            # Mocked HTTP tests
├── docs/
│   ├── voice-id.md                   # REST API reference
│   └── superpowers/                  # Design specs & implementation plans
└── asset_combine/                    # WAV files for voiceprint registration
```

**Key classes:**
- `WespeakerDeep(base_url, api_key)` — main client, async-free. Methods: `enroll()`, `recognize()`, `recognize_multi_pcm()`, `load_templates()`
- `RecognitionResult` — NamedTuple with `is_recognized`, `confidence`, `name`

**Architecture pattern:** The library sends audio files/PCM to `voiceprint-api` via REST, parses JSON responses, and returns backward-compatible dicts/NamedTuples.

## Tech Stack

- Python 3.10+, uv package management
- `requests>=2.28` — HTTP client
- `soundfile>=0.13` — PCM→WAV temp files for `recognize_multi_pcm()`
- pytest 8+ for testing (mocked HTTP via `unittest.mock`)

## Common Commands

```bash
# Install (no ML dependencies, fast)
uv sync

# Run all tests
uv run pytest

# Single test file
uv run pytest tests/wespeaker_deep_edge/test_client.py -v

# Single test
uv run pytest tests/wespeaker_deep_edge/test_client.py::test_enroll_success -v

# Coverage (threshold 25%)
uv run pytest --cov --cov-fail-under=25

# Format
uv run black . && uv run isort .

# Build wheel
uv build --wheel

# Package for deployment (Docker/Jetson)
tar -czf wespeaker-deep-edge-docker-v0.2.0.tar.gz \
    --exclude="__pycache__" --exclude="*.pyc" \
    src/wespeaker_deep_edge/ pyproject.toml README.md

# Install tar on remote
pip install --no-deps wespeaker-deep-edge-docker-v0.2.0.tar.gz
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_ID_URL` | `http://192.168.5.9:8005` | voice-id API server |
| `VOICE_ID_KEY` | `""` | API Bearer token |

## CLI Usage

```bash
# Register a speaker (calls POST /voiceprint/register)
uv run python -m wespeaker_deep_edge enroll john audio.wav

# Recognize (default: compare against index 0 = john)
uv run python -m wespeaker_deep_edge recognize test.wav

# Recognize with specific candidates
uv run python -m wespeaker_deep_edge recognize test.wav "john,frank,albert"

# List built-in speakers
uv run python -m wespeaker_deep_edge list-voiceprints

# Specify URL and key inline
uv run python -m wespeaker_deep_edge --url http://10.0.0.1:8005 --key my-token recognize test.wav
```

## Python API

```python
from wespeaker_deep_edge import WespeakerDeep, RecognitionResult

client = WespeakerDeep(base_url="http://192.168.5.9:8005", api_key="your-key")
client.enroll("speaker.wav", "voice_john.pkl")

client.load_templates(indices=[0, 1, 2])
result = client.recognize("test.wav")
# → {"is_recognized": True, "confidence": 0.85, "threshold": 0.2}

result2 = client.recognize_multi_pcm(pcm_array, sample_rate=16000)
# → RecognitionResult(is_recognized=True, confidence=0.85, name="john")
```

## Built-in Voiceprint Index

| Index | speaker_id |
|-------|-----------|
| 0 | john |
| 1 | frank |
| 2 | michael |
| 3 | qingqing |
| 4 | xixi |
| 5 | zhong |
| 6 | angle |
| 7 | albert |

## REST API (voiceprint-api)

See `docs/voice-id.md` for full API reference. Key endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/voiceprint/register` | Register (multipart: speaker_id + file) |
| POST | `/voiceprint/identify` | Identify (multipart: speaker_ids + file) |
| GET | `/voiceprint/health?key=<token>` | Health check |
| DELETE | `/voiceprint/{speaker_id}` | Delete voiceprint |

## Deployment

The library is deployed as a tar to Jetson (192.168.5.10) and installed in conda envs:

```bash
scp wespeaker-deep-edge-docker-v0.2.0.tar.gz jetson@192.168.5.10:/tmp/
ssh jetson@192.168.5.10
source /home/jetson/miniforge3/etc/profile.d/conda.sh
conda activate voice-id-02
pip install --no-deps /tmp/wespeaker-deep-edge-docker-v0.2.0.tar.gz
```

## Testing Strategy

Tests use `unittest.mock.patch` to intercept HTTP calls — no real API server needed:

```python
@patch("wespeaker_deep_edge.client.requests.post")
def test_recognize_success(mock_post, client):
    mock_post.return_value.json.return_value = {"speaker_id": "john", "score": 0.85}
    result = client.recognize("test.wav")
    assert result["is_recognized"] is True
```

## API Method Signatures

- `WespeakerDeep(base_url=None, api_key=None)` — env var fallbacks
- `enroll(audio_path, pk_path="voice.pkl")` → `{"ok": bool, "msg": str}`
- `recognize(audio_path, voiceprint=None)` → `{"is_recognized": bool, "confidence": float, "threshold": float}`
- `recognize_multi_pcm(pcm, sample_rate=16000)` → `RecognitionResult`
- `load_templates(indices=None, files=None)` → caches speaker IDs internally

## Version History

- **0.2.0** — HTTP client refactor (current). Pure API client, no ML deps.
- 0.1.x — Legacy ONNX + PyTorch dual engine (deleted).
