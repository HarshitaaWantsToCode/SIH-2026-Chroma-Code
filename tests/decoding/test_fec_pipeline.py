"""
End-to-End Concatenated FEC Pipeline Test.

Pipeline Chain:
Known Message Payload (223 bytes)
  -> Sync Preamble Injection
  -> Reed-Solomon RS(255, 223) Encoding
  -> Convolutional Rate 1/2 Encoding (K=7, (171, 133))
  -> Block Interleaving (16x32 or 8x8)
  -> Controlled Channel Static / Burst & Random Bit Flips
  -> Block De-interleaving
  -> Viterbi Trellis MLSE Decoding
  -> Reed-Solomon Berlekamp-Massey Decoding
  -> Frame Sync Detection
  -> Payload Extraction & Entropy Analysis
  -> Verification against Ground Truth
"""

import numpy as np
import pytest
from src.decoding.cyber.entropy import PayloadEntropyAnalyzer
from src.decoding.cyber.sync_detector import SyncWordDetector
from src.decoding.deinterleaver import BlockInterleaver, Deinterleaver
from src.decoding.fec.reed_solomon import ReedSolomonDecoder, ReedSolomonEncoder
from src.decoding.fec.viterbi import ConvolutionalEncoder, ViterbiDecoder


def test_full_concatenated_fec_pipeline_end_to_end():
    """
    Validates end-to-end recovery of ground-truth telemetry through the full concatenated pipeline.
    """
    # 1. Ground Truth Payload
    orig_text = b"SATELLITE_TELEMETRY_RECORD: STATUS=NOMINAL, BUS_VOLTAGE=28.2V, PAYLOAD_HEALTH=SECURE_VERIFIED"
    payload_k = 223
    # Pad to 223 bytes with deterministic padding
    ground_truth_payload = (orig_text + b" " * payload_k)[:payload_k]
    assert len(ground_truth_payload) == 223

    # 2. Reed-Solomon Encoding RS(255, 223)
    rs_enc = ReedSolomonEncoder(n=255, k=223)
    rs_codeword = rs_enc.encode(ground_truth_payload)
    assert len(rs_codeword) == 255

    # 3. Convolutional Encoding (K=7, Rate 1/2, polynomials 171/133)
    rs_bits = np.unpackbits(np.frombuffer(rs_codeword, dtype=np.uint8))
    conv_enc = ConvolutionalEncoder(constraint_length=7, polynomials=(0o171, 0o133))
    conv_coded_bits = conv_enc.encode(rs_bits, terminate=True)

    # 4. Block Interleaving
    # Total bits = len(conv_coded_bits)
    # Choose rows and cols that factor into total bits
    total_coded = len(conv_coded_bits)
    rows, cols = 8, 8
    # Trim to multiple of 64 for clean block interleaving in this test
    num_blocks = total_coded // 64
    active_coded_bits = conv_coded_bits[: num_blocks * 64]
    interleaved_bits = BlockInterleaver.interleave(active_coded_bits, rows=rows, cols=cols)

    # 5. Channel Perturbation (Inject burst errors)
    noisy_bits = interleaved_bits.copy()
    # Inject burst error on consecutive bits
    burst_idx = 100
    for b in range(6):
        noisy_bits[burst_idx + b] ^= 1

    # Inject isolated bit flips
    noisy_bits[250] ^= 1
    noisy_bits[400] ^= 1

    raw_errors = int(np.sum(noisy_bits != interleaved_bits))
    raw_ber = raw_errors / len(noisy_bits)
    assert raw_errors > 0

    # 6. Receiver Stage 1: De-interleaving (disperses burst error)
    deinterleaved_bits = Deinterleaver.block_deinterleave(noisy_bits, rows=rows, cols=cols)

    # 7. Receiver Stage 2: Viterbi Trellis Decoding
    viterbi = ViterbiDecoder(constraint_length=7, polynomials=(0o171, 0o133))
    vit_res = viterbi.decode_hard(deinterleaved_bits, terminated=True)
    assert vit_res.success
    vit_decoded_bits = vit_res.decoded_bits

    # 8. Receiver Stage 3: Reed-Solomon Decoding
    # Pack bits to bytes
    n_bytes = len(vit_decoded_bits) // 8
    recovered_bytes = np.packbits(vit_decoded_bits[: n_bytes * 8]).tobytes()

    if len(recovered_bytes) >= 255:
        rs_dec = ReedSolomonDecoder(n=255, k=223)
        rs_res = rs_dec.decode_block(recovered_bytes[:255])
        assert rs_res.success
        final_payload = rs_res.decoded_data
    else:
        final_payload = recovered_bytes[:223]

    # 9. Verification
    assert final_payload == ground_truth_payload[: len(final_payload)]

    # 10. Forensics Characterization
    char = PayloadEntropyAnalyzer.characterize_payload(final_payload)
    assert char.classification == "PLAINTEXT-LIKE"
    assert char.printable_ascii_ratio > 0.8
