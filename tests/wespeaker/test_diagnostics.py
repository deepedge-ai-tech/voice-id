"""Tests for diagnostics module."""

from __future__ import annotations

import time

import numpy as np
import torch

from src.wespeaker.diagnostics import PerformanceMetrics, RegistrationDiagnostics


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
