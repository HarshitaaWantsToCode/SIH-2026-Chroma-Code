"""
Tests for Viterbi K=7 Rate 1/2 Convolutional Decoder.
"""

import numpy as np
import pytest
from src.decoding.fec.viterbi import ConvolutionalEncoder, ViterbiDecoder


def test_viterbi_zero_error_recovery():
    """Validates perfect bit recovery under zero noise/errors."""
    encoder = ConvolutionalEncoder(constraint_length=7, polynomials=(0o171, 0o133))
    decoder = ViterbiDecoder(constraint_length=7, polynomials=(0o171, 0o133))

    rng = np.random.default_rng(42)
    message_lengths = [16, 32, 64, 128]

    for m_len in message_lengths:
        orig_bits = rng.integers(0, 2, size=m_len, dtype=np.uint8)
        
        # 1. Terminated mode
        coded_term = encoder.encode(orig_bits, terminate=True)
        res_term = decoder.decode_hard(coded_term, terminated=True)
        assert res_term.success
        assert np.array_equal(res_term.decoded_bits, orig_bits), f"Failed terminated recovery for len {m_len}"

        # 2. Unterminated / streaming mode
        coded_untermed = encoder.encode(orig_bits, terminate=False)
        res_untermed = decoder.decode_hard(coded_untermed, terminated=False)
        assert res_untermed.success
        assert np.array_equal(res_untermed.decoded_bits, orig_bits), f"Failed unterminated recovery for len {m_len}"


def test_viterbi_noise_correction_and_ber_gain():
    """Validates that Viterbi decoder corrects bit errors and improves BER."""
    encoder = ConvolutionalEncoder()
    decoder = ViterbiDecoder()

    rng = np.random.default_rng(100)
    num_bits = 200
    orig_bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    coded = encoder.encode(orig_bits, terminate=True)
    noisy_coded = coded.copy()

    # Inject isolated bit flips across the coded stream (e.g., 5% BER)
    error_indices = [5, 23, 51, 88, 120, 165, 210, 280]
    for idx in error_indices:
        if idx < len(noisy_coded):
            noisy_coded[idx] ^= 1

    pre_errors = len(error_indices)
    raw_ber = pre_errors / len(coded)

    res = decoder.decode_hard(noisy_coded, terminated=True)
    post_errors = int(np.sum(res.decoded_bits != orig_bits))
    post_ber = post_errors / len(orig_bits)

    assert res.success
    assert post_ber < raw_ber, f"Expected BER reduction, got raw={raw_ber}, post={post_ber}"
    assert post_errors == 0, "Viterbi should correct isolated bit flips completely"


def test_viterbi_soft_decision():
    """Validates soft-decision decoding using signed LLR / confidence values."""
    encoder = ConvolutionalEncoder()
    decoder = ViterbiDecoder()

    rng = np.random.default_rng(200)
    orig_bits = rng.integers(0, 2, size=64, dtype=np.uint8)
    coded = encoder.encode(orig_bits, terminate=True)

    # Convert binary {0, 1} to ideal BPSK levels {+1.0, -1.0} and add AWGN
    ideal_soft = np.where(coded == 0, 1.0, -1.0)
    noise = rng.normal(0, 0.4, size=len(ideal_soft))
    noisy_soft = ideal_soft + noise

    res = decoder.decode_soft(noisy_soft, terminated=True)
    assert res.success
    assert np.array_equal(res.decoded_bits, orig_bits)


def test_viterbi_invalid_input_rejection():
    """Ensures odd lengths or invalid values trigger explicit errors."""
    decoder = ViterbiDecoder()
    with pytest.raises(ValueError):
        decoder.decode_hard(np.array([1, 0, 1], dtype=np.uint8))  # Odd length
    with pytest.raises(ValueError):
        decoder.decode_hard(np.array([1, 2], dtype=np.uint8))     # Non-binary
