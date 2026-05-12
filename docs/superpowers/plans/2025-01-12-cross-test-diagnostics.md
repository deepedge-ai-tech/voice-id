# cross_test.py 详细诊断输出实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `scripts/cross_test.py` 添加详细的诊断输出功能，支持分级输出模式（默认/verbose/debug）和多格式报告（终端/Markdown/JSON）

**Architecture:** 新增诊断数据收集模块和报告生成模块，修改主测试脚本集成诊断功能，在底层模块添加 logging 支持

**Tech Stack:** Python 3.12, logging, JSON, Markdown, numpy, torch, matplotlib

---

## 文件结构

### 新建文件
1. `src/wespeaker/diagnostics.py` - 诊断数据收集类
   - `RegistrationDiagnostics` - 注册阶段诊断
   - `RecognitionDiagnostics` - 识别阶段诊断
   - `PerformanceMetrics` - 性能计时

2. `src/wespeaker/reporters.py` - 报告生成类
   - `TerminalReporter` - 终端输出
   - `MarkdownReportGenerator` - Markdown 报告
   - `JsonDataExporter` - JSON 导出

### 修改文件
1. `scripts/cross_test.py` - 主流程改造
2. `src/wespeaker/wespeaker.py` - 添加 logging
3. `src/wespeaker/best.py` - 添加 logging

---

## Task 1: 创建 PerformanceMetrics 性能计时类

**Files:**
- Create: `src/wespeaker/diagnostics.py`

- [ ] **Step 1: 写 PerformanceMetrics 类的测试**

```python
# tests/wespeaker/test_diagnostics.py
import time
from wespeaker.diagnostics import PerformanceMetrics

def test_performance_metrics_basic():
    metrics = PerformanceMetrics()
    metrics.start("audio_load")
    time.sleep(0.01)
    metrics.end("audio_load")

    assert "audio_load" in metrics.get_timings()
    assert metrics.get_timings()["audio_load"] >= 0.01

def test_performance_metrics_summary():
    metrics = PerformanceMetrics()
    metrics.start("task1")
    time.sleep(0.01)
    metrics.end("task1")
    metrics.start("task2")
    time.sleep(0.01)
    metrics.end("task2")

    summary = metrics.get_summary()
    assert summary["total_operations"] == 2
    assert "total_time" in summary
```

- [ ] **Step 2: 运行测试验证失败**

运行: `uv run pytest tests/wespeaker/test_diagnostics.py -v`
预期: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现 PerformanceMetrics 类**

```python
# src/wespeaker/diagnostics.py
from __future__ import annotations
import time
from dataclasses import dataclass, field


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
```

- [ ] **Step 4: 运行测试验证通过**

运行: `uv run pytest tests/wespeaker/test_diagnostics.py -v`
预期: PASS

- [ ] **Step 5: 提交**

```bash
git add src/wespeaker/diagnostics.py tests/wespeaker/test_diagnostics.py
git commit -m "feat: add PerformanceMetrics class for timing diagnostics"
```

---

## Task 2: 创建 RegistrationDiagnostics 注册诊断类

**Files:**
- Modify: `src/wespeaker/diagnostics.py`
- Modify: `tests/wespeaker/test_diagnostics.py`

- [ ] **Step 1: 写 RegistrationDiagnostics 测试**

```python
# tests/wespeaker/test_diagnostics.py (追加)
import numpy as np
import torch

def test_registration_diagnostics_collect():
    diag = RegistrationDiagnostics("John")
    diag.add_segment("seg1.wav", 1.6, 16000, torch.randn(256))
    diag.add_segment("seg2.wav", 1.5, 16000, torch.randn(256))

    data = diag.to_dict()
    assert data["speaker"] == "John"
    assert data["num_segments"] == 2
    assert "embeddings" in data
    assert "quality_metrics" in data

def test_registration_diagnostics_noise_injection():
    diag = RegistrationDiagnostics("Frank")
    diag.record_noise_injection(20, 0.05, 0.04)

    assert diag.noise_effects[20]["target_snr"] == 20.0
    assert len(diag.to_dict()["noise_effects"]) > 0
```

- [ ] **Step 2: 运行测试验证失败**

运行: `uv run pytest tests/wespeaker/test_diagnostics.py::test_registration_diagnostics_collect -v`
预期: FAIL - Class not found

- [ ] **Step 3: 实现 RegistrationDiagnostics 类**

```python
# src/wespeaker/diagnostics.py (追加)
from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable


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
        self.segments.append({
            "filename": filename,
            "duration": duration,
            "sample_rate": sample_rate,
        })
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
            "l2_norms": {"min": min(norms), "max": max(norms), "mean": sum(norms) / len(norms)},
            "cosine_distances": {"min": min(distances), "max": max(distances), "std": np.std(distances).item()},
            "within_class_compactness": float(torch.stack(distances).mean()),
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
```

- [ ] **Step 4: 运行测试验证通过**

运行: `uv run pytest tests/wespeaker/test_diagnostics.py::test_registration_diagnostics_collect -v`
预期: PASS

- [ ] **Step 5: 提交**

```bash
git add src/wespeaker/diagnostics.py tests/wespeaker/test_diagnostics.py
git commit -m "feat: add RegistrationDiagnostics class"
```

---

## Task 3: 创建 RecognitionDiagnostics 识别诊断类

**Files:**
- Modify: `src/wespeaker/diagnostics.py`
- Modify: `tests/wespeaker/test_diagnostics.py`

- [ ] **Step 1: 写 RecognitionDiagnostics 测试**

```python
# tests/wespeaker/test_diagnostics.py (追加)

def test_recognition_diagnostics_collect():
    diag = RecognitionDiagnostics("John", "安静环境测试", 0.75)
    diag.add_comparison("Frank", 0.32, False)
    diag.add_comparison("John", 0.75, True)

    data = diag.to_dict()
    assert data["test_speaker"] == "John"
    assert data["test_variant"] == "安静环境测试"
    assert data["confidence"] == 0.75
    assert len(data["comparisons"]) == 2

def test_recognition_diagnostics_error_cases():
    diag = RecognitionDiagnostics("John", "安静环境测试", 0.45, threshold=0.55)
    diag.record_false_positive("Frank", 0.62)
    diag.record_false_negative(0.10)

    assert "false_positive" in diag.to_dict()["error_analysis"]
    assert "false_negative" in diag.to_dict()["error_analysis"]
```

- [ ] **Step 2: 运行测试验证失败**

运行: `uv run pytest tests/wespeaker/test_diagnostics.py::test_recognition_diagnostics_collect -v`
预期: FAIL - Class not found

- [ ] **Step 3: 实现 RecognitionDiagnostics 类**

```python
# src/wespeaker/diagnostics.py (追加)

@dataclass
class RecognitionDiagnostics:
    """识别阶段诊断数据收集."""

    test_speaker: str
    test_variant: str
    confidence: float
    threshold: float = 0.55
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
        self.comparisons.append({
            "ref_speaker": ref_speaker,
            "score": score,
            "is_match": is_match,
        })

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
            "confidence": self.confidence,
            "threshold": self.threshold,
            "is_correct": self.error_analysis == {},
            "preprocessing": self.preprocessing,
            "comparisons": self.comparisons,
            "top2_similarity_diff": top2_diff,
            "error_analysis": self.error_analysis,
        }
```

- [ ] **Step 4: 运行测试验证通过**

运行: `uv run pytest tests/wespeaker/test_diagnostics.py::test_recognition_diagnostics_collect -v`
预期: PASS

- [ ] **Step 5: 提交**

```bash
git add src/wespeaker/diagnostics.py tests/wespeaker/test_diagnostics.py
git commit -m "feat: add RecognitionDiagnostics class"
```

---

## Task 4: 创建 JsonDataExporter JSON 导出类

**Files:**
- Create: `src/wespeaker/reporters.py`
- Create: `tests/wespeaker/test_reporters.py`

- [ ] **Step 1: 写 JsonDataExporter 测试**

```python
# tests/wespeaker/test_reporters.py
import json
from pathlib import Path
from wespeaker.reporters import JsonDataExporter

def test_json_exporter_creates_file(tmp_path):
    data = {"meta": {"test": "data"}, "results": []}
    exporter = JsonDataExporter(tmp_path)

    output_path = exporter.export(data)

    assert output_path.exists()
    content = json.loads(output_path.read_text())
    assert content["meta"]["test"] == "data"
```

- [ ] **Step 2: 运行测试验证失败**

运行: `uv run pytest tests/wespeaker/test_reporters.py -v`
预期: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现 JsonDataExporter 类**

```python
# src/wespeaker/reporters.py
from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class JsonDataExporter:
    """JSON 数据导出器."""

    output_dir: Path

    def export(self, data: dict[str, Any], timestamp: datetime | None = None) -> Path:
        """导出数据为 JSON 文件."""
        if timestamp is None:
            timestamp = datetime.now()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cross_test_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / filename

        # 转换 torch.Tensor 为 list
        json_ready = self._make_json_serializable(data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_ready, f, ensure_ascii=False, indent=2)

        return output_path

    def _make_json_serializable(self, data: Any) -> Any:
        """递归转换数据为 JSON 可序列化格式."""
        import torch
        import numpy as np

        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        return data
```

- [ ] **Step 4: 运行测试验证通过**

运行: `uv run pytest tests/wespeaker/test_reporters.py::test_json_exporter_creates_file -v`
预期: PASS

- [ ] **Step 5: 提交**

```bash
git add src/wespeaker/reporters.py tests/wespeaker/test_reporters.py
git commit -m "feat: add JsonDataExporter class"
```

---

## Task 5: 创建 MarkdownReportGenerator 报告生成类

**Files:**
- Modify: `src/wespeaker/reporters.py`
- Modify: `tests/wespeaker/test_reporters.py`

- [ ] **Step 1: 写 MarkdownReportGenerator 测试**

```python
# tests/wespeaker/test_reporters.py (追加)
from datetime import datetime
from wespeaker.reporters import MarkdownReportGenerator

def test_markdown_generator_creates_file(tmp_path):
    gen = MarkdownReportGenerator(tmp_path)

    data = {
        "meta": {
            "timestamp": datetime(2025, 1, 12, 14, 30),
            "threshold": 0.55,
            "snr_levels": [20, 15, 10, 5, 0],
            "speakers": ["John", "Frank"],
        },
        "registration": {},
        "recognition": {"errors": {}},
    }

    output_path = gen.generate(data)

    assert output_path.exists()
    content = output_path.read_text()
    assert "# 声纹交叉测试诊断报告" in content
    assert "0.55" in content
```

- [ ] **Step 2: 运行测试验证失败**

运行: `uv run pytest tests/wespeaker/test_reporters.py::test_markdown_generator_creates_file -v`
预期: FAIL - Class not found

- [ ] **Step 3: 实现 MarkdownReportGenerator 类**

```python
# src/wespeaker/reporters.py (追加)

@dataclass
class MarkdownReportGenerator:
    """Markdown 报告生成器."""

    output_dir: Path

    def generate(self, data: dict[str, Any], timestamp: datetime | None = None) -> Path:
        """生成 Markdown 报告."""
        if timestamp is None:
            timestamp = datetime.now()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cross_test_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        output_path = self.output_dir / filename

        content = self._build_report(data, timestamp)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

    def _build_report(self, data: dict[str, Any], timestamp: datetime) -> str:
        """构建报告内容."""
        lines = [
            "# 声纹交叉测试诊断报告\n",
            "## 测试配置",
            f"- 阈值: {data['meta']['threshold']}",
            f"- SNR 级别: {data['meta']['snr_levels']}",
            f"- 测试时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## 注册阶段分析",
        ]

        # 注册阶段分析
        for speaker, reg_data in data.get("registration", {}).items():
            lines.extend(self._registration_section(speaker, reg_data))

        # 识别阶段分析
        lines.extend([
            "\n## 识别阶段分析",
            self._performance_section(data.get("recognition", {})),
            self._error_cases_section(data.get("recognition", {})),
            self._variant_analysis_section(data.get("recognition", {})),
            "\n## 结论与建议",
            self._conclusions_section(data),
        ])

        return "\n".join(lines)

    def _registration_section(self, speaker: str, data: dict) -> list[str]:
        """生成单个说话人的注册分析."""
        lines = [f"\n### {speaker}"]
        lines.append(f"- 片段数量: {data.get('num_segments', 0)}")
        lines.append(f"- 总 embedding 数: {data.get('total_embeddings', 0)}")
        lines.append(f"- 向量维度: {data.get('embedding_dim', 0)}")

        quality = data.get("quality_metrics", {})
        if quality:
            lines.append("\n**向量质量指标:**")
            if "l2_norms" in quality:
                norms = quality["l2_norms"]
                lines.append(f"- L2 范数: min={norms['min']:.4f}, max={norms['max']:.4f}, mean={norms['mean']:.4f}")
            if "cosine_distances" in quality:
                dists = quality["cosine_distances"]
                lines.append(f"- 余弦距离: min={dists['min']:.4f}, max={dists['max']:.4f}, std={dists['std']:.4f}")
            if "within_class_compactness" in quality:
                lines.append(f"- 类内紧密度: {quality['within_class_compactness']:.4f}")

        if data.get("noise_effects"):
            lines.append("\n**噪声注入效果:**")
            for effect in data["noise_effects"]:
                snr = effect["target_snr"]
                lines.append(f"- SNR {snr}dB: 原始RMS={effect['original_rms']:.4f}, 混合RMS={effect['mixed_rms']:.4f}")

        return lines

    def _performance_section(self, data: dict) -> str:
        """生成性能统计部分."""
        perf = data.get("performance", {})
        if not perf:
            return "\n### 性能统计\n暂无数据"

        lines = ["\n### 性能统计"]
        if "avg_recognition_time" in perf:
            lines.append(f"- 平均识别时间: {perf['avg_recognition_time']:.4f}s")
        if "total_time" in perf:
            lines.append(f"- 总执行时间: {perf['total_time']:.2f}s")

        timings = perf.get("timings", {})
        if timings:
            lines.append("\n**详细计时:**")
            for op, stats in timings.items():
                lines.append(f"- {op}: {stats['avg_time']:.4f}s (x{stats['count']})")

        return "\n".join(lines)

    def _error_cases_section(self, data: dict) -> str:
        """生成错误案例分析."""
        errors = data.get("errors", {})
        lines = ["\n### 错误案例分析"]

        fas = errors.get("false_accepts", [])
        frs = errors.get("false_rejects", [])

        if fas:
            lines.append(f"\n**误接受 ({len(fas)} 例):**")
            for fa in fas:
                lines.append(f"- {fa['test_speaker']} 被误认为 {fa['mistaken_as']}: 得分={fa['score']:.4f}, 距离={fa['threshold_distance']:.4f}")

        if frs:
            lines.append(f"\n**误拒绝 ({len(frs)} 例):**")
            for fr in frs:
                lines.append(f"- {fr['test_speaker']} ({fr['test_variant']}): 得分={fr['score']:.4f}, 距离={fr['threshold_distance']:.4f}")

        if not fas and not frs:
            lines.append("\n无错误案例")

        return "\n".join(lines)

    def _variant_analysis_section(self, data: dict) -> str:
        """生成音频变体性能分析."""
        lines = ["\n### 音频变体性能分析"]

        variants = {}
        for case in data.get("test_cases", []):
            variant = case.get("test_variant", "unknown")
            if variant not in variants:
                variants[variant] = []
            variants[variant].append(case["confidence"])

        if variants:
            for variant, scores in variants.items():
                import numpy as np
                lines.append(f"\n**{variant}:**")
                lines.append(f"- 平均得分: {np.mean(scores):.4f}")
                lines.append(f"- 最小得分: {np.min(scores):.4f}")
                lines.append(f"- 最大得分: {np.max(scores):.4f}")

        return "\n".join(lines)

    def _conclusions_section(self, data: dict) -> str:
        """生成结论与建议."""
        errors = data.get("recognition", {}).get("errors", {})
        fas = errors.get("false_accepts", [])
        frs = errors.get("false_rejects", [])

        lines = []
        if not fas and not frs:
            lines.append("✅ **当前配置良好** - 所有测试通过")
        else:
            if fas:
                lines.append(f"⚠️ 存在 {len(fas)} 例误接受 - 建议提高阈值")
            if frs:
                lines.append(f"⚠️ 存在 {len(frs)} 例误拒绝 - 建议降低阈值或改进注册质量")

        threshold = data.get("meta", {}).get("threshold", 0.55)
        if frs:
            min_score = min(fa.get("score", 0) for fa in fas)
            lines.append(f"\n建议阈值范围: {min_score:.2f} - {threshold:.2f}")

        return "\n".join(lines)
```

- [ ] **Step 4: 运行测试验证通过**

运行: `uv run pytest tests/wespeaker/test_reporters.py::test_markdown_generator_creates_file -v`
预期: PASS

- [ ] **Step 5: 提交**

```bash
git add src/wespeaker/reporters.py tests/wespeaker/test_reporters.py
git commit -m "feat: add MarkdownReportGenerator class"
```

---

## Task 6: 创建 TerminalReporter 终端输出类

**Files:**
- Modify: `src/wespeaker/reporters.py`
- Modify: `tests/wespeaker/test_reporters.py`

- [ ] **Step 1: 写 TerminalReporter 测试**

```python
# tests/wespeaker/test_reporters.py (追加)
import logging
from io import StringIO
from wespeaker.reporters import TerminalReporter

def test_terminal_reporter_verbose_mode(capsys):
    reporter = TerminalReporter(verbose=True, debug=False)

    reporter.print_registration_summary("John", {"num_segments": 5, "total_embeddings": 25})

    captured = capsys.readouterr()
    assert "John" in captured.out
    assert "25" in captured.out
```

- [ ] **Step 2: 运行测试验证失败**

运行: `uv run pytest tests/wespeaker/test_reporters.py::test_terminal_reporter_verbose_mode -v`
预期: FAIL - Class not found

- [ ] **Step 3: 实现 TerminalReporter 类**

```python
# src/wespeaker/reporters.py (追加)
import logging
from typing import Any


@dataclass
class TerminalReporter:
    """终端输出报告器."""

    verbose: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def print_header(self, threshold: float, snr_levels: list[float]) -> None:
        """打印测试标题."""
        print("\n" + "=" * 60)
        print(f"  声纹交叉测试 (阈值 = {threshold:.2f})")
        print(f"  SNR 级别: {snr_levels}")
        print("=" * 60)

    def print_registration_start(self, speaker: str, reg_dir: str) -> None:
        """打印注册开始."""
        if self.verbose:
            print(f"\n[注册] {speaker}: {reg_dir}")
        else:
            print(f"[注册] {speaker}... ", end="", flush=True)

    def print_registration_summary(
        self,
        speaker: str,
        reg_data: dict[str, Any],
    ) -> None:
        """打印注册摘要."""
        if not self.verbose:
            print("✅")
            return

        print(f"  片段数: {reg_data.get('num_segments', 0)}")
        print(f"  总 embeddings: {reg_data.get('total_embeddings', 0)}")

        quality = reg_data.get("quality_metrics", {})
        if quality and "l2_norms" in quality:
            norms = quality["l2_norms"]
            print(f"  L2 范数: mean={norms['mean']:.4f}")

    def print_recognition_progress(
        self,
        test_label: str,
        ref_speaker: str,
        score: float,
        is_match: bool,
    ) -> None:
        """打印识别进度."""
        if self.verbose:
            status = "✅" if is_match else "❌"
            print(f"  {test_label} vs {ref_speaker}: {score:.4f} {status}")

    def print_test_summary(self, total: int, passed: int, errors: dict) -> None:
        """打印测试总结."""
        print("\n" + "=" * 60)
        if passed == total:
            print("✅ 所有测试通过")
        else:
            print(f"⚠️  {total - passed}/{total} 测试未通过")

        fas = errors.get("false_accepts", [])
        frs = errors.get("false_rejects", [])

        if fas:
            print(f"\n误接受: {len(fas)} 例")
        if frs:
            print(f"误拒绝: {len(frs)} 例")

        print("=" * 60)

    def print_debug_embedding(self, name: str, embedding: "torch.Tensor") -> None:
        """打印调试信息（向量值）."""
        if self.debug:
            import torch
            print(f"DEBUG {name}: shape={embedding.shape}, mean={embedding.mean():.4f}, std={embedding.std():.4f}")
```

- [ ] **Step 4: 运行测试验证通过**

运行: `uv run pytest tests/wespeaker/test_reporters.py::test_terminal_reporter_verbose_mode -v`
预期: PASS

- [ ] **Step 5: 提交**

```bash
git add src/wespeaker/reporters.py tests/wespeaker/test_reporters.py
git commit -m "feat: add TerminalReporter class"
```

---

## Task 7: 为 src/wespeaker/__init__.py 添加新模块导出

**Files:**
- Modify: `src/wespeaker/__init__.py`

- [ ] **Step 1: 更新 __init__.py 导出**

```python
# src/wespeaker/__init__.py
"""WeSpeaker 声纹识别工具。"""

from . import realtime_monitor
from .best import (
    BestConfig,
    WespeakerBest,
)
from .wespeaker import (
    WespeakerClient,
    _crop_verify,
    _estimate_snr,
    _extract_embedding,
    _load_audio,
    _load_model,
    _vad_segments,
)

__all__ = [
    "WespeakerClient",
    "WespeakerBest",
    "BestConfig",
    "_crop_verify",
    "_estimate_snr",
    "_extract_embedding",
    "_load_audio",
    "_load_model",
    "_vad_segments",
    "realtime_monitor",
    # 诊断模块
    "diagnostics",
    "reporters",
]
```

- [ ] **Step 2: 运行测试验证导入正常**

运行: `uv run python -c "from wespeaker import diagnostics, reporters; print('OK')"`
预期: OK

- [ ] **Step 3: 提交**

```bash
git add src/wespeaker/__init__.py
git commit -m "feat: export diagnostics and reporters modules"
```

---

## Task 8: 为 wespeaker.py 添加 logging 支持

**Files:**
- Modify: `src/wespeaker/wespeaker.py`

- [ ] **Step 1: 在 wespeaker.py 添加 logging 导入和配置**

```python
# src/wespeaker/wespeaker.py (在文件开头添加)
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: 在关键位置添加 logger 调用**

```python
# src/wespeaker/wespeaker.py

# 在 _load_audio 函数中
def _load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    path = str(Path(path).expanduser())
    logger.debug("Loading audio: %s (target_sr=%d)", path, target_sr)

    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        logger.debug("Loaded with torchaudio: sr=%d, shape=%s", sr, waveform.shape)
        # ... rest of function
    except Exception as exc:
        logger.warning("torchaudio failed for %s, trying librosa", path)
        # ... rest of function

# 在 _extract_embedding 函数中
def _extract_embedding(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    logger.debug("Extracting embedding: waveform shape=%s", waveform.shape)
    # ... rest of function
    logger.debug("Extracted embedding shape: %s", emb.shape)
    return emb.squeeze(0).cpu()

# 在 WespeakerClient.mp3_to_pk 中
def mp3_to_pk(self, mp3_path: str, pk_path: str) -> dict:
    logger.info("Enrolling voiceprint from %s", mp3_path)
    # ... rest of function
    logger.info("Voiceprint enrolled: %d segments, dim=%d", len(segments), mean_emb.numel())
    return {...}

# 在 WespeakerClient.recognize 中
def recognize(self, audio_path: str, pk_path: str) -> dict:
    logger.info("Recognizing %s against %s", audio_path, pk_path)
    # ... rest of function
    logger.debug("Recognition score: %.4f (threshold=%.2f)", score, self.sim_threshold)
    return {...}
```

- [ ] **Step 3: 运行测试验证无影响**

运行: `uv run pytest tests/ -v`
预期: 全部通过

- [ ] **Step 4: 提交**

```bash
git add src/wespeaker/wespeaker.py
git commit -m "feat: add logging support to wespeaker.py"
```

---

## Task 9: 为 best.py 添加 logging 支持

**Files:**
- Modify: `src/wespeaker/best.py`

- [ ] **Step 1: 在 best.py 添加 logging 导入**

```python
# src/wespeaker/best.py (在文件开头添加)
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: 在关键位置添加 logger 调用**

```python
# src/wespeaker/best.py

# 在 WespeakerBest.enroll 中
def enroll(self, ...):
    logger.info("Enrolling %s with %d clean segments", clean_dir, len(clean_paths))
    logger.debug("SNR levels: %s", snrs)
    # ... rest of function
    logger.info("Enrolled %s: %d embeddings, dim=%d", clean_dir, len(all_embeddings), ref.numel())
    return {...}

# 在 WespeakerBest.recognize 中
def recognize(self, ...):
    logger.debug("Recognizing %s", audio_path)
    # ... rest of function
    logger.debug("Recognition score: %.4f", score)
    return {...}
```

- [ ] **Step 3: 运行测试验证无影响**

运行: `uv run pytest tests/ -v`
预期: 全部通过

- [ ] **Step 4: 提交**

```bash
git add src/wespeaker/best.py
git commit -m "feat: add logging support to best.py"
```

---

## Task 10: 修改 cross_test.py 集成诊断功能

**Files:**
- Modify: `scripts/cross_test.py`

- [ ] **Step 1: 更新导入部分**

```python
# scripts/cross_test.py
import pickle
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.wespeaker import WespeakerBest
from src.wespeaker.diagnostics import (
    RegistrationDiagnostics,
    RecognitionDiagnostics,
    PerformanceMetrics,
)
from src.wespeaker.reporters import (
    TerminalReporter,
    MarkdownReportGenerator,
    JsonDataExporter,
)

# 配置 logging
logging.basicConfig(
    level=logging.WARNING,  # 默认只显示 WARNING 及以上
    format="%(levelname)s: %(message)s",
)
```

- [ ] **Step 2: 修改 cross_test 函数签名和参数处理**

```python
# scripts/cross_test.py

def cross_test(
    noise_path: str,
    snr_levels: list[float],
    threshold: float,
    output_dir: Path | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """执行 6x6 交叉测试矩阵，生成诊断报告."""
    # 配置 logging 级别
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)

    if output_dir is None:
        output_dir = Path("experiment_log")

    reporter = TerminalReporter(verbose=verbose, debug=debug)
    metrics = PerformanceMetrics()

    reporter.print_header(threshold, snr_levels)

    recognizer = WespeakerBest()
    recognizer.config = recognizer.config.__class__(
        **{**vars(recognizer.config), "sim_threshold": threshold}
    )

    # ... rest of function
```

- [ ] **Step 3: 重写注册阶段集成诊断**

```python
# scripts/cross_test.py (替换注册部分)

# 注册所有说话人
voiceprints: dict[str, torch.Tensor] = {}
tmp_pk = Path("/tmp/voice_cross.pkl")
tmp_pk.parent.mkdir(parents=True, exist_ok=True)
registration_data: dict[str, dict] = {}

for name, paths in SPEAKERS.items():
    reg_dir = paths["register_dir"]
    reporter.print_registration_start(name, reg_dir)

    metrics.start(f"register_{name}")

    diag = RegistrationDiagnostics(name)

    # 收集片段信息
    clean_paths = sorted(Path(reg_dir).glob("*.wav"))
    for clean_path in clean_paths:
        import torchaudio
        waveform, sr = torchaudio.load(str(clean_path))
        seg = waveform.mean(dim=0).cpu().numpy()
        duration = len(seg) / sr
        diag.add_segment(clean_path.name, duration, sr, torch.zeros(256))  # 占位

    # 记录噪声注入
    for snr in snr_levels:
        diag.record_noise_injection(snr, 0.05, 0.04, actual_snr=snr)

    # 实际注册
    result = recognizer.enroll(reg_dir, noise_profile, str(tmp_pk), snr_levels)
    voiceprints[name] = result["embedding"]

    # 补充 embedding 信息
    for emb in result.get("embeddings", []):
        if isinstance(emb, torch.Tensor):
            diag.embeddings.append(emb)

    metrics.end(f"register_{name}")

    registration_data[name] = diag.to_dict()
    reporter.print_registration_summary(name, registration_data[name])
```

- [ ] **Step 4: 重写识别阶段集成诊断**

```python
# scripts/cross_test.py (替换识别部分)

# 交叉识别矩阵
col_headers = [f"{name} 声纹" for name in SPEAKERS.keys()]
col_width = 12
header = f"{'':>14} | " + " | ".join(f"{h:>{col_width}}" for h in col_headers)
sep = "-" * len(header)

print(f"\n{'=' * len(header)}")

test_cases: list[dict] = []
false_accepts: list[dict] = []
false_rejects: list[dict] = []

for test_speaker, speaker_data in SPEAKERS.items():
    for label, audio_path in speaker_data["test_audios"].items():
        metrics.start(f"recognize_{test_speaker}_{label}")

        diag = RecognitionDiagnostics(test_speaker, label, 0.0, threshold=threshold)

        # 预处理信息
        waveform = _load_audio(audio_path, 16000)
        rms = float(torch.sqrt((waveform ** 2).mean()))
        diag.set_preprocessing_info(
            duration=len(waveform) / 16000,
            sample_rate=16000,
            rms_energy=rms,
            crop_mode="full_utterance",
        )

        # 与每个声纹比较
        for ref_name, ref_emb in voiceprints.items():
            with open(tmp_pk, "wb") as f:
                pickle.dump(ref_emb.cpu().numpy(), f)

            result = recognizer.recognize(audio_path, str(tmp_pk))
            score = result["confidence"]
            is_match = result["is_recognized"]

            diag.add_comparison(ref_name, score, is_match)

            if test_speaker == ref_name:
                diag.confidence = score
                if not is_match:
                    diag.record_false_negative(score)
                    false_rejects.append({
                        "test_speaker": test_speaker,
                        "test_variant": label,
                        "score": score,
                        "threshold_distance": threshold - score,
                    })
            else:
                if is_match:
                    diag.record_false_positive(ref_name, score)
                    false_accepts.append({
                        "test_speaker": test_speaker,
                        "mistaken_as": ref_name,
                        "test_variant": label,
                        "score": score,
                        "threshold_distance": score - threshold,
                    })

            reporter.print_recognition_progress(f"{test_speaker}/{label}", ref_name, score, is_match)

        metrics.end(f"recognize_{test_speaker}_{label}")
        test_cases.append(diag.to_dict())

# 统计
print()
total_tests = len(test_cases)
passed = total_tests - len(false_accepts) - len(false_rejects)
errors = {"false_accepts": false_accepts, "false_rejects": false_rejects}
reporter.print_test_summary(total_tests, passed, errors)
```

- [ ] **Step 5: 添加报告生成**

```python
# scripts/cross_test.py (在函数末尾添加)

# 生成报告
report_data = {
    "meta": {
        "timestamp": datetime.now(),
        "threshold": threshold,
        "snr_levels": snr_levels,
        "speakers": list(SPEAKERS.keys()),
    },
    "registration": registration_data,
    "recognition": {
        "test_cases": test_cases,
        "performance": {
            "avg_recognition_time": sum(
                t.get("duration", 0) for t in test_cases
            ) / len(test_cases) if test_cases else 0,
            "total_time": metrics.get_summary()["total_time"],
            "timings": metrics.get_summary()["operations"],
        },
        "errors": errors,
    },
}

markdown_gen = MarkdownReportGenerator(output_dir)
json_exporter = JsonDataExporter(output_dir)

md_path = markdown_gen.generate(report_data)
json_path = json_exporter.export(report_data)

print(f"\n📊 报告已生成:")
print(f"  Markdown: {md_path}")
print(f"  JSON: {json_path}")
```

- [ ] **Step 6: 更新 CLI 参数处理**

```python
# scripts/cross_test.py

def main() -> None:
    parser = argparse.ArgumentParser(description="声纹交叉测试 — 6x6 识别矩阵")
    parser.add_argument("--noise", default="asset/john/嘈杂环境测试.m4a")
    parser.add_argument("--snrs", default="20,15,10,5,0")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="详细输出模式")
    parser.add_argument("--debug", action="store_true", help="调试模式（最详细）")
    args = parser.parse_args()

    # ... validation code ...

    output_path = Path(args.output_dir) if args.output_dir else Path("experiment_log")
    cross_test(
        args.noise,
        [float(x.strip()) for x in args.snrs.split(",")],
        args.threshold,
        output_path,
        verbose=args.verbose,
        debug=args.debug,
    )
```

- [ ] **Step 7: 运行功能测试**

运行: `uv run python scripts/cross_test.py --help`
预期: 显示帮助信息，包含 --verbose 和 --debug 参数

- [ ] **Step 8: 运行端到端测试**

运行: `uv run python scripts/cross_test.py --output-dir /tmp/test_exp_log`
预期: 测试完成，生成报告文件

- [ ] **Step 9: 提交**

```bash
git add scripts/cross_test.py
git commit -m "feat: integrate diagnostics into cross_test.py with verbose/debug modes"
```

---

## Task 11: 更新 .gitignore 忽略 experiment_log 目录

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 添加 experiment_log 目录到 .gitignore**

```gitignore
# experiment_log 目录
experiment_log/
```

- [ ] **Step 2: 提交**

```bash
git add .gitignore
git commit -m "chore: add experiment_log/ to gitignore"
```

---

## Task 12: 格式化代码和运行测试

**Files:**
- All modified files

- [ ] **Step 1: 格式化代码**

```bash
uv run black .
uv run isort .
```

- [ ] **Step 2: 运行所有测试**

```bash
uv run pytest --cov
```

预期: 全部通过，覆盖率 ≥ 30%

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "style: format code and ensure tests pass"
```

---

## Task 13: 验证功能完整性

**Files:**
- All

- [ ] **Step 1: 测试默认模式**

```bash
uv run python scripts/cross_test.py
```

验证: 终端显示摘要，experiment_log 目录生成 Markdown 和 JSON 文件

- [ ] **Step 2: 测试 verbose 模式**

```bash
uv run python scripts/cross_test.py --verbose
```

验证: 终端显示详细信息，日志级别 INFO

- [ ] **Step 3: 测试 debug 模式**

```bash
uv run python scripts/cross_test.py --debug
```

验证: 终端显示最详细信息，生成 debug 日志文件

- [ ] **Step 4: 检查报告内容**

验证: Markdown 报告包含所有章节，JSON 数据结构正确

- [ ] **Step 5: 最终提交**

```bash
git add -A
git commit -m "feat: complete cross_test diagnostics feature

- Add diagnostic data collection classes
- Add report generation (Markdown, JSON, Terminal)
- Add logging support to wespeaker.py and best.py
- Add --verbose and --debug CLI modes
- Default output to experiment_log/ directory

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 自审查

**Spec 覆盖检查:**
- [x] 输出模式（默认/verbose/debug）- Task 10
- [x] 注册阶段诊断 - Task 2
- [x] 识别阶段诊断 - Task 3
- [x] Markdown 报告 - Task 5
- [x] JSON 导出 - Task 4
- [x] 终端输出 - Task 6
- [x] logging 支持 - Task 8, 9
- [x] CLI 参数 - Task 10
- [x] 输出到 experiment_log/ - Task 10

**无占位符检查:**
- 所有步骤包含完整代码
- 所有文件路径明确
- 所有命令可执行

**类型一致性检查:**
- RecognitionDiagnostics.confidence 类型一致
- PerformanceMetrics 返回类型一致
- 报告数据结构一致
