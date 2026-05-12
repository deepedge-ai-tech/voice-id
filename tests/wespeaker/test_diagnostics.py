"""Tests for diagnostics module."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from src.wespeaker.diagnostics import (
    PerformanceMetrics,
    RecognitionDiagnostics,
    RegistrationDiagnostics,
)


def test_performance_metrics_basic():
    """Test basic timing functionality."""
    metrics = PerformanceMetrics()
    metrics.start("audio_load")
    time.sleep(0.01)
    elapsed = metrics.end("audio_load")

    assert "audio_load" in metrics.get_timings()
    assert metrics.get_timings()["audio_load"] >= 0.01
    assert elapsed >= 0.01


def test_performance_metrics_summary():
    """Test summary statistics."""
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
    assert summary["total_time"] >= 0.02
    assert "operations" in summary
    assert "task1" in summary["operations"]
    assert "task2" in summary["operations"]


def test_performance_metrics_multiple_calls():
    """Test multiple calls to same operation."""
    metrics = PerformanceMetrics()
    metrics.start("op")
    time.sleep(0.01)
    metrics.end("op")
    metrics.start("op")
    time.sleep(0.01)
    metrics.end("op")

    timings = metrics.get_timings()
    assert timings["op"] >= 0.02

    summary = metrics.get_summary()
    assert summary["operations"]["op"]["count"] == 2
    assert summary["operations"]["op"]["avg_time"] >= 0.01


def test_performance_metrics_end_without_start():
    """Test ending operation without starting it."""
    metrics = PerformanceMetrics()
    elapsed = metrics.end("nonexistent")

    assert elapsed == 0.0
    assert "nonexistent" not in metrics.get_timings()


def test_performance_metrics_get_timings_isolation():
    """Test that get_timings returns a copy, not the internal dict."""
    metrics = PerformanceMetrics()
    metrics.start("op")
    time.sleep(0.01)
    metrics.end("op")

    timings = metrics.get_timings()
    timings["op"] = 999.0  # Modify returned dict

    # Original should be unchanged
    assert metrics.get_timings()["op"] < 999.0


def test_registration_diagnostics_collect():
    """Test RegistrationDiagnostics data collection."""
    diag = RegistrationDiagnostics("John")
    diag.add_segment("seg1.wav", 1.6, 16000, torch.randn(256))
    diag.add_segment("seg2.wav", 1.5, 16000, torch.randn(256))

    data = diag.to_dict()
    assert data["speaker"] == "John"
    assert data["num_segments"] == 2
    assert "segments" in data
    assert "quality_metrics" in data


def test_registration_diagnostics_noise_injection():
    """Test RegistrationDiagnostics noise injection recording."""
    diag = RegistrationDiagnostics("Frank")
    diag.record_noise_injection(20, 0.05, 0.04)

    assert "snr_20" in diag.noise_effects
    assert diag.noise_effects["snr_20"]["target_snr"] == 20.0
    assert len(diag.to_dict()["noise_effects"]) > 0


def test_registration_diagnostics_quality_metrics():
    """Test RegistrationDiagnostics quality metrics calculation."""
    diag = RegistrationDiagnostics("Alice")

    # Create some embeddings with similar values (simulating same speaker)
    base_emb = torch.randn(256)
    for i in range(3):
        # Add small variation to simulate same speaker
        emb = base_emb + torch.randn(256) * 0.01
        diag.add_segment(f"seg{i}.wav", 1.5, 16000, emb)

    metrics = diag.get_quality_metrics()
    assert "l2_norms" in metrics
    assert "cosine_distances" in metrics
    assert "within_class_compactness" in metrics
    assert "min" in metrics["l2_norms"]
    assert "max" in metrics["l2_norms"]
    assert "mean" in metrics["l2_norms"]
    assert "min" in metrics["cosine_distances"]
    assert "max" in metrics["cosine_distances"]
    assert "std" in metrics["cosine_distances"]


def test_registration_diagnostics_empty():
    """Test RegistrationDiagnostics with no data."""
    diag = RegistrationDiagnostics("Bob")
    data = diag.to_dict()

    assert data["speaker"] == "Bob"
    assert data["num_segments"] == 0
    assert data["embedding_dim"] == 0
    assert data["total_embeddings"] == 0

    metrics = diag.get_quality_metrics()
    assert metrics == {}


def test_registration_diagnostics_to_dict_structure():
    """Test RegistrationDiagnostics to_dict output structure."""
    diag = RegistrationDiagnostics("Charlie")
    diag.add_segment("test.wav", 2.0, 16000, torch.randn(256))
    diag.record_noise_injection(10, 0.1, 0.08, actual_snr=9.5)

    data = diag.to_dict()

    # Verify all expected keys
    assert "speaker" in data
    assert "num_segments" in data
    assert "segments" in data
    assert "embedding_dim" in data
    assert "total_embeddings" in data
    assert "quality_metrics" in data
    assert "noise_effects" in data

    # Verify segment structure
    assert len(data["segments"]) == 1
    assert data["segments"][0]["filename"] == "test.wav"
    assert data["segments"][0]["duration"] == 2.0
    assert data["segments"][0]["sample_rate"] == 16000

    # Verify noise effects structure
    assert len(data["noise_effects"]) == 1
    noise_effect = data["noise_effects"][0]
    assert noise_effect["target_snr"] == 10
    assert noise_effect["original_rms"] == 0.1
    assert noise_effect["mixed_rms"] == 0.08
    assert noise_effect["actual_snr"] == 9.5


def test_recognition_diagnostics_collect():
    """Test RecognitionDiagnostics data collection."""
    diag = RecognitionDiagnostics("John", "安静环境测试", 0.75)
    diag.add_comparison("Frank", 0.32, False)
    diag.add_comparison("John", 0.75, True)

    data = diag.to_dict()
    assert data["test_speaker"] == "John"
    assert data["test_variant"] == "安静环境测试"
    assert data["confidence"] == 0.75
    assert len(data["comparisons"]) == 2


def test_recognition_diagnostics_error_cases():
    """Test RecognitionDiagnostics error case recording."""
    diag = RecognitionDiagnostics("John", "安静环境测试", 0.45, threshold=0.55)
    diag.record_false_positive("Frank", 0.62)
    diag.record_false_negative(0.10)

    assert "false_positive" in diag.to_dict()["error_analysis"]
    assert "false_negative" in diag.to_dict()["error_analysis"]


def test_recognition_diagnostics_preprocessing():
    """Test RecognitionDiagnostics preprocessing info."""
    diag = RecognitionDiagnostics("Alice", "嘈杂环境测试", 0.68)
    diag.set_preprocessing_info(
        duration=5.2,
        sample_rate=16000,
        rms_energy=0.15,
        vad_segments=3,
        crop_mode="full_utterance",
    )

    data = diag.to_dict()
    assert data["preprocessing"]["duration_sec"] == 5.2
    assert data["preprocessing"]["sample_rate"] == 16000
    assert data["preprocessing"]["rms_energy"] == 0.15
    assert data["preprocessing"]["vad_segments"] == 3
    assert data["preprocessing"]["crop_mode"] == "full_utterance"


def test_recognition_diagnostics_top2_diff():
    """Test RecognitionDiagnostics Top-2 similarity difference calculation."""
    diag = RecognitionDiagnostics("Bob", "测试", 0.80)
    diag.add_comparison("Bob", 0.80, True)
    diag.add_comparison("Alice", 0.45, False)
    diag.add_comparison("Charlie", 0.35, False)

    data = diag.to_dict()
    assert "top2_similarity_diff" in data
    assert data["top2_similarity_diff"] == pytest.approx(0.35)  # 0.80 - 0.45


def test_recognition_diagnostics_single_comparison():
    """Test RecognitionDiagnostics with single comparison."""
    diag = RecognitionDiagnostics("Single", "测试", 0.70)
    diag.add_comparison("Single", 0.70, True)

    data = diag.to_dict()
    assert data["top2_similarity_diff"] == 0.0


def test_recognition_diagnostics_default_values():
    """Test RecognitionDiagnostics default values."""
    diag = RecognitionDiagnostics("Test", "variant", 0.60)

    assert diag.threshold == 0.55
    assert diag.duration == 0.0
    assert diag.sample_rate == 16000
    assert diag.rms_energy == 0.0
    assert diag.comparisons == []
    assert diag.preprocessing == {}
    assert diag.error_analysis == {}


def test_recognition_diagnostics_error_analysis_details():
    """Test RecognitionDiagnostics error analysis details."""
    diag = RecognitionDiagnostics("John", "测试", 0.40, threshold=0.55)

    # Record false positive
    diag.record_false_positive("Frank", 0.62)

    # Record false negative
    diag.record_false_negative(0.30)

    data = diag.to_dict()
    error_analysis = data["error_analysis"]

    # Check false positive details
    assert error_analysis["false_positive"]["mistaken_as"] == "Frank"
    assert error_analysis["false_positive"]["score"] == 0.62
    assert error_analysis["false_positive"]["threshold_distance"] == pytest.approx(
        0.07
    )  # 0.62 - 0.55

    # Check false negative details
    assert error_analysis["false_negative"]["score"] == 0.30
    assert error_analysis["false_negative"]["threshold_distance"] == pytest.approx(
        0.25
    )  # 0.55 - 0.30


def test_recognition_diagnostics_is_correct_flag():
    """Test RecognitionDiagnostics is_correct flag."""
    # Correct case - no errors
    diag_correct = RecognitionDiagnostics("John", "测试", 0.75)
    diag_correct.add_comparison("John", 0.75, True)
    assert diag_correct.to_dict()["is_correct"] is True

    # Error case - false positive
    diag_error = RecognitionDiagnostics("John", "测试", 0.62)
    diag_error.record_false_positive("Frank", 0.62)
    assert diag_error.to_dict()["is_correct"] is False
