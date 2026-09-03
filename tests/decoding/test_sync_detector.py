"""
Tests for Frame Synchronization and Sliding Hamming Distance Detection.
"""

import numpy as np
import pytest
from src.decoding.cyber.sync_detector import SyncWordDetector


def test_sync_detector_exact_match():
    """Validates exact synchronization header location and 100% confidence."""
    sync_word = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.uint8)  # 16-bit
    rng = np.random.default_rng(42)

    stream = rng.integers(0, 2, size=300, dtype=np.uint8)
    expected_pos = 85
    stream[expected_pos : expected_pos + len(sync_word)] = sync_word

    res = SyncWordDetector.detect_sync_detailed(stream, sync_word, max_bit_errors=0)
    assert res.sync_detected
    assert res.first_sync_index == expected_pos
    assert res.min_hamming_distance == 0
    assert res.best_confidence == 1.0


def test_sync_detector_noisy_tolerance():
    """Validates detection under 1, 2, and 3 bit errors with proportional confidence."""
    sync_word = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.uint8)  # 16-bit
    stream = np.zeros(200, dtype=np.uint8)

    pos = 40
    stream[pos : pos + 16] = sync_word
    # Corrupt 2 bits
    stream[pos + 2] ^= 1
    stream[pos + 7] ^= 1

    res = SyncWordDetector.detect_sync_detailed(stream, sync_word, max_bit_errors=2)
    assert res.sync_detected
    assert res.first_sync_index == pos
    assert res.min_hamming_distance == 2
    assert res.best_confidence == (16 - 2) / 16  # 0.875


def test_sync_detector_frame_extraction():
    """Validates extraction of frame boundaries and payloads."""
    sync_word = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)  # 8 bits
    # Stream with two frames of 40 bits each
    frame1 = np.concatenate([sync_word, np.ones(32, dtype=np.uint8)])
    frame2 = np.concatenate([sync_word, np.zeros(32, dtype=np.uint8)])
    stream = np.concatenate([frame1, frame2])

    frames = SyncWordDetector.extract_frames(stream, sync_word, max_bit_errors=0, frame_length=40)
    assert len(frames) == 2
    assert frames[0].frame_start == 0
    assert frames[0].frame_end == 40
    assert np.all(frames[0].payload_bits == 1)

    assert frames[1].frame_start == 40
    assert frames[1].frame_end == 80
    assert np.all(frames[1].payload_bits == 0)
