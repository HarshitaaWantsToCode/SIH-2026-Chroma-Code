"""
Tests for Shannon Information Entropy and Sliding-Window Profiling.
"""

import numpy as np
import pytest
from src.decoding.cyber.entropy import PayloadEntropyAnalyzer


def test_entropy_extreme_bounds():
    """Validates 0.0 entropy for constant streams and ~8.0 for uniform byte distributions."""
    # 1. All zeros -> H = 0.0
    zeros = bytes([0] * 500)
    h_zero = PayloadEntropyAnalyzer.calculate_shannon_entropy(zeros)
    assert h_zero == 0.0

    # 2. Perfect uniform distribution over 256 byte values -> H = 8.0
    uniform = bytes(list(range(256)) * 4)
    h_uniform = PayloadEntropyAnalyzer.calculate_shannon_entropy(uniform)
    assert abs(h_uniform - 8.0) < 1e-6


def test_sliding_window_entropy():
    """Validates sliding-window statistics across concatenated low/high entropy segments."""
    low_seg = bytes([0] * 128)
    rng = np.random.default_rng(42)
    high_seg = rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()

    combined = low_seg + high_seg
    profile = PayloadEntropyAnalyzer.profile_sliding_window(combined, window_size=32, step_size=16)

    assert profile.min_entropy == 0.0
    assert profile.max_entropy > 4.5
    assert len(profile.windows) > 0
    assert len(profile.entropy_values) == len(profile.windows)
