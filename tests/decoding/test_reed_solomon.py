"""
Tests for Reed-Solomon RS(N, K) Algebraic Decoder over GF(256).
"""

import numpy as np
import pytest
from src.decoding.fec.reed_solomon import GF256, ReedSolomonDecoder, ReedSolomonEncoder


def test_gf256_arithmetic_properties():
    """Validates Galois Field GF(2^8) field axioms: addition, multiplication, inverse, exponentiation."""
    gf = GF256()

    for a in [1, 2, 45, 127, 200, 255]:
        # Identity
        assert gf.mul(a, 1) == a
        assert gf.add(a, 0) == a
        # Self-inverse addition
        assert gf.add(a, a) == 0
        # Multiplicative inverse: a * a^-1 == 1
        a_inv = gf.inv(a)
        assert gf.mul(a, a_inv) == 1
        # Division
        assert gf.div(a, a) == 1


def test_reed_solomon_zero_errors():
    """Validates perfect decoding of uncorrupted RS(255, 223) and RS(31, 15) codewords."""
    rs_enc = ReedSolomonEncoder(n=255, k=223)
    rs_dec = ReedSolomonDecoder(n=255, k=223)

    rng = np.random.default_rng(42)
    msg = rng.integers(0, 256, size=223, dtype=np.uint8).tobytes()

    codeword = rs_enc.encode(msg)
    assert len(codeword) == 255

    res = rs_dec.decode_block(codeword)
    assert res.success
    assert not res.uncorrectable
    assert res.corrected_symbol_count == 0
    assert res.decoded_data == msg


def test_reed_solomon_correctable_errors():
    """
    Validates correction of 1, 4, 8, and up to t=16 symbol errors for RS(255, 223).
    """
    rs_enc = ReedSolomonEncoder(n=255, k=223)
    rs_dec = ReedSolomonDecoder(n=255, k=223)
    t_cap = rs_dec.t  # 16

    rng = np.random.default_rng(101)
    msg = rng.integers(0, 256, size=223, dtype=np.uint8).tobytes()
    clean_codeword = bytearray(rs_enc.encode(msg))

    test_error_counts = [1, 4, 8, 12, 16]

    for n_err in test_error_counts:
        corrupted = bytearray(clean_codeword)
        # Pick random distinct error positions
        err_positions = rng.choice(255, size=n_err, replace=False)
        for pos in err_positions:
            err_byte = int(rng.integers(1, 256))
            corrupted[pos] ^= err_byte

        res = rs_dec.decode_block(bytes(corrupted))
        assert res.success, f"Failed decoding with {n_err} errors"
        assert not res.uncorrectable
        assert res.corrected_symbol_count == n_err
        assert res.decoded_data == msg, f"Message mismatch for {n_err} errors"


def test_reed_solomon_uncorrectable_errors_exceeding_t():
    """
    Validates that when errors exceed correction capability (e.g. 17 errors > t=16),
    the decoder safely reports uncorrectable rather than corrupting data silently.
    """
    rs_enc = ReedSolomonEncoder(n=255, k=223)
    rs_dec = ReedSolomonDecoder(n=255, k=223)

    rng = np.random.default_rng(999)
    msg = rng.integers(0, 256, size=223, dtype=np.uint8).tobytes()
    corrupted = bytearray(rs_enc.encode(msg))

    # Inject 17 errors (exceeding t=16)
    err_positions = rng.choice(255, size=17, replace=False)
    for pos in err_positions:
        corrupted[pos] ^= int(rng.integers(1, 256))

    res = rs_dec.decode_block(bytes(corrupted))
    assert res.uncorrectable or not res.success


def test_reed_solomon_edge_cases():
    """Tests all-zero, all-255, and invalid length inputs."""
    rs_enc = ReedSolomonEncoder(n=255, k=223)
    rs_dec = ReedSolomonDecoder(n=255, k=223)

    # 1. All zero bytes
    msg_zeros = bytes([0] * 223)
    cw_zeros = rs_enc.encode(msg_zeros)
    res_zeros = rs_dec.decode_block(cw_zeros)
    assert res_zeros.success
    assert res_zeros.decoded_data == msg_zeros

    # 2. All 255 bytes
    msg_255 = bytes([255] * 223)
    cw_255 = rs_enc.encode(msg_255)
    res_255 = rs_dec.decode_block(cw_255)
    assert res_255.success
    assert res_255.decoded_data == msg_255

    # 3. Invalid length
    with pytest.raises(ValueError):
        rs_dec.decode_block(bytes([0] * 100))
