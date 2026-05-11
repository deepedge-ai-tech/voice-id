"""
WespeakerClient — 独立的 WeSpeaker 声纹注册与识别工具。

零项目依赖，仅需安装: torch, torchaudio, numpy, audiomentations(可选)

用法:
    client = WespeakerClient()
    client.mp3_to_pk("voice.mp3", "voice.pkl")
    result = client.recognize("voice2.mp3", "voice.pkl")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  音频加载
# --------------------------------------------------------------------------- #


def _load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """读取音频文件 → 单声道 → 目标采样率 → float32 [-1,1] waveform."""
    path = str(Path(path).expanduser())
    # 优先 torchaudio（支持 mp3/wav/ogg，需 ffmpeg）
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.mean(dim=0)  # mono
        return waveform
    except Exception:
        pass
    # fallback: librosa
    try:
        import librosa

        arr, _ = librosa.load(path, sr=target_sr, mono=True)
        return torch.from_numpy(arr)
    except Exception as exc:
        raise RuntimeError(f"无法加载音频 {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
#  模型加载 (pyannote.audio 后端)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str) -> torch.nn.Module:
    """加载 WeSpeaker ResNet34 模型，缓存单例."""
    try:
        from pyannote.audio import Model
    except ImportError as exc:
        raise ImportError("未安装 pyannote.audio: pip install pyannote.audio") from exc

    # 兼容旧版 torch.load(weights_only=False)
    _real = torch.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return _real(*a, **kw)

    torch.load = _load  # type: ignore[assignment]

    dev = torch.device(device)
    model = Model.from_pretrained(model_path)
    model.eval()
    model.to(dev)
    return model


def _extract_embedding(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """从 waveform 提取 256 维 embedding."""
    device = next(model.parameters()).device
    with torch.no_grad():
        x = waveform.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,T)
        emb = model(x)  # (1, 256)
        return emb.squeeze(0).cpu()


# --------------------------------------------------------------------------- #
#  噪声增强 (可选)
# --------------------------------------------------------------------------- #


class _NoiseAugmentor:
    def __init__(
        self,
        sample_rate: int = 16000,
        augment_ratio: float = 0.6,
        noise_dir: str = "",
        seed: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.augment_ratio = augment_ratio
        import random

        self._rand = random.Random(seed)

        try:
            from audiomentations import (
                AddBackgroundNoise,
                AddGaussianSNR,
                AddShortNoises,
            )
        except ImportError:
            self._available = False
            return
        self._available = True
        noise_path = noise_dir if Path(noise_dir).is_dir() else None
        has_noise = bool(noise_path and any(Path(noise_path).rglob("*.wav")))
        self._gaussian = AddGaussianSNR(min_snr_db=5, max_snr_db=20, p=1.0)
        self._background = (
            AddBackgroundNoise(sounds_path=noise_path, min_snr_db=5, max_snr_db=20, p=1.0)
            if has_noise
            else None
        )
        self._short = (
            AddShortNoises(sounds_path=noise_path, min_snr_db=3, max_snr_db=18, p=1.0)
            if has_noise
            else None
        )

    def augment(self, segments: list[np.ndarray]) -> list[np.ndarray]:
        if not self._available or self.augment_ratio <= 0:
            return [np.asarray(s, dtype=np.float32) for s in segments]
        n = len(segments)
        k = max(0, min(int(round(n * self.augment_ratio)), n))
        aug_idx = set(self._rand.sample(range(n), k=k)) if k > 0 else set()
        out = []
        for i, seg in enumerate(segments):
            arr = np.asarray(seg, dtype=np.float32)
            if i in aug_idx:
                choice = np.random.choice(["background", "short", "gaussian"], p=[0.5, 0.2, 0.3])
                if choice == "background" and self._background:
                    arr = self._background(samples=arr, sample_rate=self.sample_rate)
                elif choice == "short" and self._short:
                    arr = self._short(samples=arr, sample_rate=self.sample_rate)
                else:
                    arr = self._gaussian(samples=arr, sample_rate=self.sample_rate)
            out.append(np.asarray(arr, dtype=np.float32))
        return out


# --------------------------------------------------------------------------- #
#  SNR 估计 & VAD
# --------------------------------------------------------------------------- #


def _estimate_snr(
    waveform: torch.Tensor, sample_rate: int = 16000, frame_ms: int = 25, hop_ms: int = 10
) -> float:
    """估计音频片段的信噪比（dB）。

    将信号分帧，以能量最低的 10% 帧作为噪声估计，
    其余帧的平均能量与噪声能量之比即为 SNR。
    """
    frame_len = int(frame_ms / 1000 * sample_rate)
    hop_len = int(hop_ms / 1000 * sample_rate)
    if frame_len < 1:
        frame_len = 160
        hop_len = 80

    frames = []
    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        seg = waveform[start : start + frame_len]
        energy = (seg**2).mean().item()
        if energy > 1e-10:
            frames.append(energy)

    if not frames:
        return 0.0

    frames.sort()
    n_noise = max(1, len(frames) // 10)
    noise_energy = np.mean(frames[:n_noise])
    signal_energy = np.mean(frames[n_noise:])

    if noise_energy < 1e-10:
        return 30.0  # cap
    return round(10 * np.log10(signal_energy / noise_energy), 2)


def _vad_segments(
    waveform: torch.Tensor,
    rms_threshold: float = 0.005,
    min_duration_ms: int = 100,
    sample_rate: int = 16000,
) -> list[torch.Tensor]:
    """基于 RMS 能量的语音活动检测，返回有语音的片段列表。

    使用自适应阈值：以 RMS 的 25 分位数作为噪声底噪估计，
    阈值 = noise_floor * adaptive_factor（默认 2.0 倍）。
    如果计算出的阈值低于固定下限，则使用固定下限。
    """
    frame_len = int(0.02 * sample_rate)  # 20ms
    hop_len = int(0.01 * sample_rate)  # 10ms
    min_samples = int(min_duration_ms / 1000 * sample_rate)

    rms_values = []
    starts = []
    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        seg = waveform[start : start + frame_len]
        rms = float(torch.sqrt((seg**2).mean()))
        rms_values.append(rms)
        starts.append(start)

    if not rms_values:
        return [waveform]

    # 自适应阈值：25 分位数 * 2.0，不低于固定下限
    sorted_rms = sorted(rms_values)
    noise_floor = sorted_rms[max(0, len(sorted_rms) // 4)]
    adaptive_threshold = max(noise_floor * 2.0, rms_threshold)

    # 合并连续的高能量帧
    segments: list[torch.Tensor] = []
    in_speech = False
    seg_start = 0
    for i, rms in enumerate(rms_values):
        if rms >= adaptive_threshold and not in_speech:
            in_speech = True
            seg_start = starts[i]
        elif (rms < adaptive_threshold or i == len(rms_values) - 1) and in_speech:
            in_speech = False
            seg_end = starts[i] + frame_len
            if seg_end - seg_start >= min_samples:
                segments.append(waveform[seg_start:seg_end])

    return segments if segments else [waveform]


# --------------------------------------------------------------------------- #
#  主类
# --------------------------------------------------------------------------- #


@dataclass
class WespeakerClient:
    """WeSpeaker 声纹注册与识别。

    所有参数可在构造时或运行时通过属性调整。
    """

    # ---- 模型 ----
    model_path: str = "./models/wespeaker"
    device: str = "cpu"

    # ---- 注册 ----
    enrollment_segment_secs: float = 1.0
    enable_augmentation: bool = True
    augment_ratio: float = 0.6
    noise_dir: str = ""
    sample_rate: int = 16000
    enable_snr_weighting: bool = True  # 注册时按 SNR 加权

    # ---- 识别 ----
    sim_threshold: float = 0.55
    verify_crop_mode: str = "full_utterance"  # full_utterance / tail_window / head_window
    verify_window_secs: float = 1.0
    verify_buffer_keep_secs: float = 8.0  # 验证 buffer 最大保留时长
    enable_vad: bool = True  # 识别时启用 VAD 去静音
    vad_rms_threshold: float = 0.005  # VAD 能量阈值

    # ---- 内部 ----
    _model: Optional[torch.nn.Module] = field(init=False, default=None, repr=False)
    _aug: Optional[_NoiseAugmentor] = field(init=False, default=None, repr=False)

    # ------------------------------------------------------------------ #
    #  公开 API
    # ------------------------------------------------------------------ #

    def mp3_to_pk(self, mp3_path: str, pk_path: str) -> dict:
        """从音频文件注册声纹 → 保存 .pkl 文件."""
        if not Path(mp3_path).is_file():
            return {"ok": False, "error": f"文件不存在: {mp3_path}"}

        self._ensure_model()
        waveform = _load_audio(mp3_path, self.sample_rate)

        # 切段
        seg_len = int(self.enrollment_segment_secs * self.sample_rate)
        if waveform.numel() < seg_len:
            return {"ok": False, "error": "音频太短，无法切分出有效片段"}
        segments = [
            waveform[i * seg_len : (i + 1) * seg_len].cpu().numpy()
            for i in range(len(waveform) // seg_len)
        ]

        # 增强
        if self.enable_augmentation:
            segments = self._augmentor().augment(segments)

        # 提取 embedding
        embeddings = []
        for seg in segments:
            t = torch.from_numpy(seg)
            emb = _extract_embedding(self._model, t)
            embeddings.append(emb)

        # 均值 + 归一化（可选 SNR 加权）
        if self.enable_snr_weighting:
            snr_values = [
                _estimate_snr(torch.from_numpy(seg), sample_rate=self.sample_rate)
                for seg in segments
            ]
            weights = torch.tensor([max(s, 0.0) for s in snr_values], dtype=torch.float32)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                stacked = torch.stack(embeddings, dim=0)
                mean_emb = F.normalize((stacked * weights.unsqueeze(1)).sum(dim=0), dim=0)
            else:
                mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)
        else:
            mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)

        # 保存
        out = Path(pk_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(mean_emb.cpu().numpy(), f)

        return {
            "ok": True,
            "num_segments": len(segments),
            "embedding_dim": mean_emb.numel(),
            "pk_path": str(out.resolve()),
        }

    def recognize(self, audio_path: str, pk_path: str) -> dict:
        """将音频与已保存的声纹比对，返回识别结果."""
        if not Path(audio_path).is_file():
            return {"is_recognized": False, "confidence": 0.0, "error": f"文件不存在: {audio_path}"}
        if not Path(pk_path).is_file():
            return {
                "is_recognized": False,
                "confidence": 0.0,
                "error": f"声纹文件不存在: {pk_path}",
            }

        self._ensure_model()

        # 加载参考声纹
        with open(pk_path, "rb") as f:
            ref = F.normalize(torch.from_numpy(np.asarray(pickle.load(f), dtype=np.float32)), dim=0)

        # 加载音频 + 限制长度 + 裁剪
        waveform = _load_audio(audio_path, self.sample_rate)
        max_samples = int(self.verify_buffer_keep_secs * self.sample_rate)
        if waveform.numel() > max_samples:
            if self.verify_crop_mode == "head_window":
                waveform = waveform[:max_samples]
            else:
                waveform = waveform[-max_samples:]

        # VAD 去静音/去纯噪声
        if self.enable_vad and self.verify_crop_mode == "full_utterance":
            speech_segs = _vad_segments(
                waveform, rms_threshold=self.vad_rms_threshold, sample_rate=self.sample_rate
            )
            if speech_segs and len(speech_segs) > 0:
                pcm = torch.cat(speech_segs)
            else:
                pcm = waveform
        else:
            pcm = _crop_verify(
                waveform, self.verify_crop_mode, self.verify_window_secs, self.sample_rate
            )
        if pcm.numel() == 0:
            return {"is_recognized": False, "confidence": 0.0, "error": "音频太短"}

        # 提取 embedding + 比对
        emb = _extract_embedding(self._model, pcm)
        emb = F.normalize(emb, dim=0)
        score = float(torch.dot(emb, ref).clamp(-1.0, 1.0).item())

        return {
            "is_recognized": score >= self.sim_threshold,
            "confidence": round(score, 4),
            "threshold": self.sim_threshold,
        }

    # ------------------------------------------------------------------ #
    #  内部
    # ------------------------------------------------------------------ #

    def enroll_mixed(self, clean_paths: list[str], noise_paths: list[str], pk_path: str) -> dict:
        """混合注册：结合 clean 和 noisy 音频生成声纹。"""
        self._ensure_model()
        all_segments: list[torch.Tensor] = []
        seg_len = int(self.enrollment_segment_secs * self.sample_rate)

        for path in clean_paths + noise_paths:
            if not Path(path).is_file():
                continue
            waveform = _load_audio(path, self.sample_rate)
            if waveform.numel() < seg_len:
                continue
            for i in range(len(waveform) // seg_len):
                all_segments.append(waveform[i * seg_len : (i + 1) * seg_len])

        if not all_segments:
            return {"ok": False, "error": "无有效音频片段"}

        embeddings = [_extract_embedding(self._model, seg) for seg in all_segments]

        if self.enable_snr_weighting:
            snr_values = [_estimate_snr(seg, sample_rate=self.sample_rate) for seg in all_segments]
            weights = torch.tensor([max(s, 0.0) for s in snr_values], dtype=torch.float32)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                mean_emb = F.normalize(
                    (torch.stack(embeddings, dim=0) * weights.unsqueeze(1)).sum(dim=0), dim=0
                )
            else:
                mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)
        else:
            mean_emb = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), dim=0)

        out = Path(pk_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(mean_emb.cpu().numpy(), f)

        return {"ok": True, "num_segments": len(all_segments), "pk_path": str(out.resolve())}

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = _load_model(self.model_path, self.device)

    def _augmentor(self) -> _NoiseAugmentor:
        if self._aug is None:
            self._aug = _NoiseAugmentor(
                sample_rate=self.sample_rate,
                augment_ratio=self.augment_ratio,
                noise_dir=self.noise_dir,
            )
        return self._aug


def _crop_verify(
    waveform: torch.Tensor, mode: str, window_secs: float, sample_rate: int
) -> torch.Tensor:
    """裁剪音频用于验证。"""
    mode = mode.lower()
    window = int(window_secs * sample_rate)
    if mode == "full_utterance":
        return waveform
    if mode == "head_window":
        return waveform[:window]
    return waveform[-window:] if waveform.numel() > window else waveform


# --------------------------------------------------------------------------- #
#  CLI 入口
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WeSpeaker 声纹注册与识别")
    sub = parser.add_subparsers(dest="cmd")

    p_reg = sub.add_parser("enroll", help="注册声纹")
    p_reg.add_argument("audio")
    p_reg.add_argument("output", nargs="?", default="voice.pkl")
    p_reg.add_argument("--model-path", default="./models/wespeaker")
    p_reg.add_argument("--device", default="cpu")
    p_reg.add_argument("--no-aug", action="store_true")

    p_rec = sub.add_parser("recognize", help="识别声纹")
    p_rec.add_argument("audio")
    p_rec.add_argument("voiceprint")
    p_rec.add_argument("--model-path", default="./models/wespeaker")
    p_rec.add_argument("--device", default="cpu")
    p_rec.add_argument("--threshold", type=float, default=0.75)

    args = parser.parse_args()

    if args.cmd == "enroll":
        client = WespeakerClient(
            model_path=args.model_path,
            device=args.device,
            enable_augmentation=not args.no_aug,
        )
        r = client.mp3_to_pk(args.audio, args.output)
        print(r)
    elif args.cmd == "recognize":
        client = WespeakerClient(
            model_path=args.model_path,
            device=args.device,
            sim_threshold=args.threshold,
        )
        r = client.recognize(args.audio, args.voiceprint)
        print(r)
    else:
        parser.print_help()
