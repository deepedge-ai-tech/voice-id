"""Performance diagnostics utilities."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


@dataclass
class PerformanceMetrics:
    """性能计时统计类."""

    _timings: dict[str, float] = field(default_factory=dict)
    _start_times: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def start(self, operation: str) -> None:
        """开始计时某个操作."""
        self._start_times[operation] = time.perf_counter()

    def end(self, operation: str) -> float:
        """结束计时某个操作，返回耗时（秒）."""
        if operation not in self._start_times:
            return 0.0
        elapsed = time.perf_counter() - self._start_times[operation]
        del self._start_times[operation]

        if operation not in self._timings:
            self._timings[operation] = 0.0
            self._counts[operation] = 0
        self._timings[operation] += elapsed
        self._counts[operation] += 1
        return elapsed

    def get_timings(self) -> dict[str, float]:
        """获取所有操作的总耗时."""
        return self._timings.copy()

    def get_summary(self) -> dict:
        """获取性能统计摘要."""
        total_time = sum(self._timings.values())
        return {
            "total_operations": len(self._timings),
            "total_time": total_time,
            "operations": {
                op: {
                    "total_time": self._timings[op],
                    "count": self._counts[op],
                    "avg_time": self._timings[op] / self._counts[op],
                }
                for op in self._timings
            },
        }


@dataclass
class RegistrationDiagnostics:
    """注册阶段诊断数据收集."""

    speaker: str
    segments: list[dict] = field(default_factory=list)
    embeddings: list[torch.Tensor] = field(default_factory=list)
    noise_effects: dict[str, dict] = field(default_factory=dict)

    def add_segment(
        self,
        filename: str,
        duration: float,
        sample_rate: int,
        embedding: torch.Tensor,
    ) -> None:
        """添加一个注册片段的信息."""
        self.segments.append(
            {
                "filename": filename,
                "duration": duration,
                "sample_rate": sample_rate,
            }
        )
        self.embeddings.append(embedding)

    def record_noise_injection(
        self,
        snr_level: float,
        original_rms: float,
        mixed_rms: float,
        actual_snr: float | None = None,
    ) -> None:
        """记录噪声注入效果."""
        key = f"snr_{snr_level}"
        if key not in self.noise_effects:
            self.noise_effects[key] = {
                "target_snr": snr_level,
                "original_rms": original_rms,
                "mixed_rms": mixed_rms,
                "actual_snr": actual_snr,
            }

    def get_quality_metrics(self) -> dict:
        """计算向量质量指标."""
        if not self.embeddings:
            return {}

        stacked = torch.stack(self.embeddings)
        mean_emb = stacked.mean(dim=0)
        mean_emb_norm = F.normalize(mean_emb.unsqueeze(0), dim=0).squeeze(0)

        # 每个 embedding 的范数
        norms = [float(emb.norm()) for emb in self.embeddings]

        # 与均值的余弦距离
        distances = []
        for emb in self.embeddings:
            emb_norm = F.normalize(emb.unsqueeze(0), dim=0).squeeze(0)
            dist = 1.0 - float(torch.dot(emb_norm, mean_emb_norm))
            distances.append(dist)

        return {
            "l2_norms": {
                "min": min(norms),
                "max": max(norms),
                "mean": sum(norms) / len(norms),
            },
            "cosine_distances": {
                "min": min(distances),
                "max": max(distances),
                "std": np.std(distances).item(),
            },
            "within_class_compactness": sum(distances) / len(distances),
        }

    def to_dict(self) -> dict:
        """导出为字典（用于 JSON 序列化）."""
        return {
            "speaker": self.speaker,
            "num_segments": len(self.segments),
            "segments": self.segments,
            "embedding_dim": self.embeddings[0].numel() if self.embeddings else 0,
            "total_embeddings": len(self.embeddings),
            "quality_metrics": self.get_quality_metrics(),
            "noise_effects": list(self.noise_effects.values()),
        }


@dataclass
class RecognitionDiagnostics:
    """识别阶段诊断数据收集."""

    test_speaker: str
    test_variant: str
    threshold: float = 0.55
    confidence: float | None = None
    duration: float = 0.0
    sample_rate: int = 16000
    rms_energy: float = 0.0
    comparisons: list[dict] = field(default_factory=list)
    preprocessing: dict = field(default_factory=dict)
    error_analysis: dict = field(default_factory=dict)

    def add_comparison(
        self,
        ref_speaker: str,
        score: float,
        is_match: bool,
    ) -> None:
        """添加与一个参考声纹的比较结果."""
        self.comparisons.append(
            {
                "ref_speaker": ref_speaker,
                "score": score,
                "is_match": is_match,
            }
        )

    def set_preprocessing_info(
        self,
        duration: float,
        sample_rate: int,
        rms_energy: float,
        vad_segments: int | None = None,
        crop_mode: str | None = None,
    ) -> None:
        """设置预处理信息."""
        self.duration = duration
        self.sample_rate = sample_rate
        self.rms_energy = rms_energy
        self.preprocessing = {
            "duration_sec": duration,
            "sample_rate": sample_rate,
            "rms_energy": rms_energy,
            "vad_segments": vad_segments,
            "crop_mode": crop_mode,
        }

    def record_false_positive(
        self,
        mistaken_speaker: str,
        score: float,
    ) -> None:
        """记录误接受案例."""
        self.error_analysis["false_positive"] = {
            "mistaken_as": mistaken_speaker,
            "score": score,
            "threshold_distance": score - self.threshold,
        }

    def record_false_negative(
        self,
        score: float,
    ) -> None:
        """记录误拒绝案例."""
        self.error_analysis["false_negative"] = {
            "score": score,
            "threshold_distance": self.threshold - score,
        }

    def to_dict(self) -> dict:
        """导出为字典."""
        # 计算 Top-2 相似度差异
        scores = [c["score"] for c in self.comparisons]
        sorted_scores = sorted(scores, reverse=True)
        top2_diff = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0

        return {
            "test_speaker": self.test_speaker,
            "test_variant": self.test_variant,
            "confidence": self.confidence if self.confidence is not None else 0.0,
            "threshold": self.threshold,
            "is_correct": self.error_analysis == {},
            "preprocessing": self.preprocessing,
            "comparisons": self.comparisons,
            "top2_similarity_diff": top2_diff,
            "error_analysis": self.error_analysis,
        }
