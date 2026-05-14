"""Data export and reporting utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class JsonDataExporter:
    """JSON 数据导出器.

    将数据导出为 JSON 文件，支持自动转换 torch.Tensor 和 numpy.ndarray
    为 JSON 可序列化格式。

    Attributes:
        output_dir: 输出目录路径

    Example:
        >>> exporter = JsonDataExporter(Path("output"))
        >>> data = {"embedding": torch.tensor([1, 2, 3])}
        >>> output_path = exporter.export(data)
    """

    output_dir: Path

    def export(self, data: dict[str, Any], timestamp: datetime | None = None) -> Path:
        """导出数据为 JSON 文件.

        Args:
            data: 要导出的数据字典
            timestamp: 可选的时间戳，用于生成文件名。默认为当前时间。

        Returns:
            导出的 JSON 文件路径

        Raises:
            OSError: 文件写入失败
        """
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
        """递归转换数据为 JSON 可序列化格式.

        支持转换以下类型：
        - torch.Tensor -> list
        - numpy.ndarray -> list
        - dict: 递归处理所有值
        - list/tuple: 递归处理所有元素

        Args:
            data: 任意数据

        Returns:
            JSON 可序列化的数据
        """
        import numpy as np
        import torch

        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        return data


@dataclass
class MarkdownReportGenerator:
    """Markdown 报告生成器.

    生成人类可读的声纹交叉测试诊断报告，包含测试配置、注册阶段分析、
    识别阶段分析、错误案例分析和结论建议。

    Attributes:
        output_dir: 输出目录路径

    Example:
        >>> gen = MarkdownReportGenerator(Path("reports"))
        >>> data = {"meta": {...}, "registration": {...}, "recognition": {...}}
        >>> report_path = gen.generate(data)
    """

    output_dir: Path

    def generate(self, data: dict[str, Any], timestamp: datetime | None = None) -> Path:
        """生成 Markdown 报告.

        Args:
            data: 包含测试结果的字典，需要包含 meta, registration, recognition 键
            timestamp: 可选的时间戳，用于生成文件名。默认为当前时间。

        Returns:
            生成的 Markdown 文件路径

        Raises:
            OSError: 文件写入失败
            KeyError: 数据字典缺少必需的键
        """
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
        """构建报告内容.

        Args:
            data: 包含测试结果的字典
            timestamp: 测试时间戳

        Returns:
            完整的 Markdown 报告内容
        """
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
        lines.extend(
            [
                "\n## 识别阶段分析",
                self._performance_section(data.get("recognition", {})),
                self._error_cases_section(data.get("recognition", {})),
                self._variant_analysis_section(data.get("recognition", {})),
                "\n## 结论与建议",
                self._conclusions_section(data),
            ]
        )

        return "\n".join(lines)

    def _registration_section(self, speaker: str, data: dict) -> list[str]:
        """生成单个说话人的注册分析.

        Args:
            speaker: 说话人名称
            data: 该说话人的注册数据

        Returns:
            Markdown 格式的注册分析文本
        """
        lines = [f"\n### {speaker}"]
        lines.append(f"- 片段数量: {data.get('num_segments', 0)}")
        lines.append(f"- 总 embedding 数: {data.get('total_embeddings', 0)}")
        lines.append(f"- 向量维度: {data.get('embedding_dim', 0)}")

        quality = data.get("quality_metrics", {})
        if quality:
            lines.append("\n**向量质量指标:**")
            if "l2_norms" in quality:
                norms = quality["l2_norms"]
                lines.append(
                    f"- L2 范数: min={norms['min']:.4f}, max={norms['max']:.4f}, mean={norms['mean']:.4f}"
                )
            if "cosine_distances" in quality:
                dists = quality["cosine_distances"]
                lines.append(
                    f"- 余弦距离: min={dists['min']:.4f}, max={dists['max']:.4f}, std={dists['std']:.4f}"
                )
            if "within_class_compactness" in quality:
                lines.append(f"- 类内紧密度: {quality['within_class_compactness']:.4f}")

        if data.get("noise_effects"):
            lines.append("\n**噪声注入效果:**")
            for effect in data["noise_effects"]:
                snr = effect["target_snr"]
                lines.append(
                    f"- SNR {snr}dB: 原始RMS={effect['original_rms']:.4f}, 混合RMS={effect['mixed_rms']:.4f}"
                )

        return lines

    def _performance_section(self, data: dict) -> str:
        """生成性能统计部分.

        Args:
            data: 识别阶段数据

        Returns:
            Markdown 格式的性能统计文本
        """
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

        # 添加VAD处理统计
        test_cases = data.get("test_cases", [])
        vad_stats = self._compute_vad_stats(test_cases)
        if vad_stats:
            lines.append("\n**VAD处理统计:**")
            lines.append(f"- 原始音频平均时长: {vad_stats['avg_original']:.2f}s")
            lines.append(f"- VAD处理后平均时长: {vad_stats['avg_vad']:.2f}s")
            lines.append(f"- 平均保留比例: {vad_stats['avg_ratio']:.1f}%")

        return "\n".join(lines)

    def _compute_vad_stats(self, test_cases: list) -> dict:
        """计算VAD处理统计数据."""
        import numpy as np

        original_durations = []
        vad_durations = []
        ratios = []

        for case in test_cases:
            prep = case.get("preprocessing", {})
            orig = prep.get("original_duration")
            vad = prep.get("vad_duration")
            if orig is not None and vad is not None and orig > 0:
                original_durations.append(orig)
                vad_durations.append(vad)
                ratios.append(vad / orig * 100)

        if not original_durations:
            return {}

        return {
            "avg_original": np.mean(original_durations),
            "avg_vad": np.mean(vad_durations),
            "avg_ratio": np.mean(ratios),
            "min_ratio": np.min(ratios) if ratios else 0,
            "max_ratio": np.max(ratios) if ratios else 0,
        }

    def _error_cases_section(self, data: dict) -> str:
        """生成错误案例分析.

        Args:
            data: 识别阶段数据

        Returns:
            Markdown 格式的错误案例分析文本
        """
        errors = data.get("errors", {})
        test_cases = data.get("test_cases", [])
        # 创建测试案例查找索引
        case_lookup = {
            (tc.get("test_speaker"), tc.get("test_variant")): tc
            for tc in test_cases
        }

        lines = ["\n### 错误案例分析"]

        fas = errors.get("false_accepts", [])
        frs = errors.get("false_rejects", [])

        if fas:
            lines.append(f"\n**误接受 ({len(fas)} 例):**")
            for fa in fas:
                lines.append(
                    f"- {fa['test_speaker']} 被误认为 {fa['mistaken_as']}: 得分={fa['score']:.4f}, 距离={fa['threshold_distance']:.4f}"
                )

        if frs:
            lines.append(f"\n**误拒绝 ({len(frs)} 例):**")
            for fr in frs:
                # 查找对应的测试案例获取VAD信息
                case_key = (fr.get("test_speaker"), fr.get("test_variant"))
                case = case_lookup.get(case_key, {})
                prep = case.get("preprocessing", {})
                orig = prep.get("original_duration")
                vad = prep.get("vad_duration")

                # 构建VAD信息字符串
                vad_info = ""
                if orig is not None and vad is not None:
                    ratio = (vad / orig * 100) if orig > 0 else 0
                    vad_info = f", 原始时长={orig:.2f}s, VAD后={vad:.2f}s (保留{ratio:.0f}%)"

                lines.append(
                    f"- {fr['test_speaker']} ({fr['test_variant']}): 得分={fr['score']:.4f}, 距离={fr['threshold_distance']:.4f}{vad_info}"
                )

        if not fas and not frs:
            lines.append("\n无错误案例")

        return "\n".join(lines)

    def _variant_analysis_section(self, data: dict) -> str:
        """生成音频变体性能分析.

        Args:
            data: 识别阶段数据

        Returns:
            Markdown 格式的音频变体分析文本
        """
        lines = ["\n### 音频变体性能分析"]

        variants = {}
        for case in data.get("test_cases", []):
            variant = case.get("test_variant", "unknown")
            if variant not in variants:
                variants[variant] = []
            variants[variant].append(case["confidence"])

        if variants:
            import numpy as np

            for variant, scores in variants.items():
                lines.append(f"\n**{variant}:**")
                lines.append(f"- 平均得分: {np.mean(scores):.4f}")
                lines.append(f"- 最小得分: {np.min(scores):.4f}")
                lines.append(f"- 最大得分: {np.max(scores):.4f}")

        return "\n".join(lines)

    def _conclusions_section(self, data: dict) -> str:
        """生成结论与建议.

        Args:
            data: 测试结果数据

        Returns:
            Markdown 格式的结论与建议文本
        """
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
        if frs and fas:
            min_score = min(fa.get("score", 0) for fa in fas)
            lines.append(f"\n建议阈值范围: {min_score:.2f} - {threshold:.2f}")

        return "\n".join(lines)


@dataclass
class TerminalReporter:
    """终端输出报告器."""

    verbose: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def print_header(self, threshold: float, snr_levels: list[float], threshold_type: str = "固定") -> None:
        """打印测试标题."""
        print("\n" + "=" * 60)
        print(f"  声纹交叉测试 ({threshold_type}阈值 = {threshold:.2f})")
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

            print(
                f"DEBUG {name}: shape={embedding.shape}, mean={embedding.mean():.4f}, std={embedding.std():.4f}"
            )
