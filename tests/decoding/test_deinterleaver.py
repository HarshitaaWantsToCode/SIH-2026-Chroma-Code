"""
Tests for Block, Convolutional, and Pseudo-Random De-interleaving Mechanisms.
"""

import numpy as np
import pytest
from src.decoding.deinterleaver import (
    BlockInterleaver,
    ConvolutionalDeinterleaver,
    ConvolutionalInterleaver,
    Deinterleaver,
)


def test_block_deinterleave_exact_roundtrip():
    """Validates exact reconstruction of original bitstream across multiple dimensions."""
    rng = np.random.default_rng(42)
    test_dimensions = [(8, 8), (4, 16), (10, 10), (7, 13)]

    for rows, cols in test_dimensions:
        block_size = rows * cols
        num_blocks = 5
        orig_bits = rng.integers(0, 2, size=num_blocks * block_size, dtype=np.uint8)

        # Transmitter: Interleave (row-write, col-read)
        interleaved = BlockInterleaver.interleave(orig_bits, rows=rows, cols=cols)

        # Receiver: De-interleave (col-write, row-read)
        recovered = Deinterleaver.block_deinterleave(interleaved, rows=rows, cols=cols)

        assert np.array_equal(orig_bits, recovered), f"Failed roundtrip for {rows}x{cols}"


def test_block_deinterleave_incomplete_modes():
    """Validates truncate and pad modes on non-multiples of block size."""
    rows, cols = 4, 4  # Block size 16
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)  # 19 bits (16 + 3)

    # 1. Truncate mode: drops trailing 3 bits
    trunc_meta = Deinterleaver.block_deinterleave_with_meta(bits, rows=rows, cols=cols, incomplete_mode="truncate")
    assert trunc_meta.output_length == 16
    assert trunc_meta.truncated_elements == 3
    assert trunc_meta.padding_added == 0

    # 2. Pad mode: pads 13 zeros to complete second 16-bit block (total 32 bits)
    pad_meta = Deinterleaver.block_deinterleave_with_meta(bits, rows=rows, cols=cols, incomplete_mode="pad", pad_value=0)
    assert pad_meta.output_length == 32
    assert pad_meta.padding_added == 13
    assert pad_meta.truncated_elements == 0


def test_block_deinterleave_validation():
    """Ensures improper parameters raise explicit ValueError."""
    with pytest.raises(ValueError):
        Deinterleaver.block_deinterleave(np.array([1, 0, 1]), rows=-1, cols=4)
    with pytest.raises(ValueError):
        Deinterleaver.block_deinterleave(np.array([1, 2, 0]), rows=2, cols=2)  # Non-binary
    with pytest.raises(ValueError):
        Deinterleaver.block_deinterleave(np.array([[1, 0], [0, 1]]), rows=2, cols=2)  # 2D array


def test_convolutional_deinterleave_exact_roundtrip():
    """Validates shift-register based convolutional interleaving and deinterleaving."""
    rng = np.random.default_rng(123)
    test_configs = [(4, 2), (6, 3), (8, 4)]

    for branches, delay_inc in test_configs:
        c_intl = ConvolutionalInterleaver(branches=branches, delay_increment=delay_inc)
        c_deint = ConvolutionalDeinterleaver(branches=branches, delay_increment=delay_inc)

        latency = c_deint.total_latency
        num_symbols = 200 + latency
        orig_symbols = rng.integers(0, 256, size=num_symbols, dtype=np.uint8)

        # Stream through interleaver -> deinterleaver
        interleaved = c_intl.process(orig_symbols)
        recovered = c_deint.process(interleaved)

        # Data after total_latency must exactly match original data from index 0
        valid_recovered = recovered[latency:]
        expected_orig = orig_symbols[: len(valid_recovered)]

        assert np.array_equal(valid_recovered, expected_orig), f"Failed for B={branches}, M={delay_inc}"


def test_pseudo_random_deinterleave_roundtrip():
    """Validates pseudo-random permutation inversion."""
    rng = np.random.default_rng(789)
    n = 64
    orig_data = rng.integers(0, 2, size=n, dtype=np.uint8)

    # Random valid permutation of [0..N-1]
    perm = rng.permutation(n)

    # Forward interleaving
    interleaved = orig_data[perm]

    # Deinterleaving via inverse permutation
    recovered = Deinterleaver.pseudo_random_deinterleave(interleaved, perm)

    assert np.array_equal(orig_data, recovered)


def test_pseudo_random_invalid_permutation():
    """Tests invalid permutation handling (duplicates, out of bounds)."""
    with pytest.raises(ValueError):
        Deinterleaver.inverse_permutation([0, 1, 1, 3])  # Duplicate 1
    with pytest.raises(ValueError):
        Deinterleaver.inverse_permutation([0, 1, 2, 5])  # Out of bounds
    with pytest.raises(ValueError):
        Deinterleaver.pseudo_random_deinterleave(np.array([1, 0, 1]), [0, 1])  # Length mismatch
