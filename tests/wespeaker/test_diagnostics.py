"""Tests for diagnostics module."""

from __future__ import annotations

import time

from src.wespeaker.diagnostics import PerformanceMetrics


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
