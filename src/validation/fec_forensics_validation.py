"""
Comprehensive FEC and Cyber Forensics Validation Suite.

Executes standalone end-to-end verification across all 17 requirements:
1. De-interleaving (Block round-trip, Convolutional delay round-trip, Pseudo-random permutation inversion)
2. Viterbi Trellis MLSE Decoder (K=7, Rate 1/2, polynomials 171/133, zero-error, noisy BER reduction, soft-decision)
3. Reed-Solomon RS(255, 223) Decoder over GF(256) (0, 1, 4, 8, 16 errors corrected, 17 errors rejected as uncorrectable)
4. Frame Synchronization (exact sync, noisy Hamming tolerance, frame extraction)
5. Shannon Information Entropy & Sliding-Window Profiling
6. Payload Forensics & Conservative Characterization (Plaintext, Structured, Compressed, High-Entropy)
7. End-to-End Concatenated Pipeline
"""

import sys
import numpy as np

from src.decoding.cyber.entropy import PayloadEntropyAnalyzer
from src.decoding.cyber.forensics_payloads import SyntheticForensicsDataset
from src.decoding.cyber.sync_detector import SyncWordDetector
from src.decoding.deinterleaver import (
    BlockInterleaver,
    ConvolutionalDeinterleaver,
    ConvolutionalInterleaver,
    Deinterleaver,
)
from src.decoding.fec.reed_solomon import ReedSolomonDecoder, ReedSolomonEncoder
from src.decoding.fec.viterbi import ConvolutionalEncoder, ViterbiDecoder


def run_fec_forensics_validation() -> bool:
    print("=" * 80)
    print("CHROMA CODE — FEC & CYBER FORENSICS VALIDATION BENCHMARK (SIH 2026)")
    print("=" * 80)

    all_passed = True

    # ---------------- 1. DE-INTERLEAVING BENCHMARK ----------------
    print("\n[STAGE 1] DE-INTERLEAVING SUBSYSTEM:")
    rng = np.random.default_rng(42)

    # 1.1 Block Deinterleaver Round-trip
    block_pass = True
    for rows, cols in [(8, 8), (4, 16), (10, 10)]:
        orig = rng.integers(0, 2, size=5 * rows * cols, dtype=np.uint8)
        inter = BlockInterleaver.interleave(orig, rows, cols)
        deint = Deinterleaver.block_deinterleave(inter, rows, cols)
        if not np.array_equal(orig, deint):
            block_pass = False
            break
    print(f"  • Block De-interleaver (8x8, 4x16, 10x10):        {'[PASS]' if block_pass else '[FAIL]'}")
    all_passed = all_passed and block_pass

    # 1.2 Convolutional Deinterleaver Round-trip
    c_intl = ConvolutionalInterleaver(branches=4, delay_increment=2)
    c_deint = ConvolutionalDeinterleaver(branches=4, delay_increment=2)
    latency = c_deint.total_latency
    orig_c = rng.integers(0, 256, size=200 + latency, dtype=np.uint8)
    int_c = c_intl.process(orig_c)
    rec_c = c_deint.process(int_c)
    conv_pass = np.array_equal(rec_c[latency:], orig_c[:len(rec_c) - latency])
    print(f"  • Convolutional De-interleaver (B=4, M=2):       {'[PASS]' if conv_pass else '[FAIL]'} (Latency: {latency} sym)")
    all_passed = all_passed and conv_pass

    # 1.3 Pseudo-Random Permutation Inversion
    p_len = 64
    orig_p = rng.integers(0, 2, size=p_len, dtype=np.uint8)
    perm = rng.permutation(p_len)
    rec_p = Deinterleaver.pseudo_random_deinterleave(orig_p[perm], perm)
    pr_pass = np.array_equal(orig_p, rec_p)
    print(f"  • Pseudo-Random De-interleaver (N=64):           {'[PASS]' if pr_pass else '[FAIL]'}")
    all_passed = all_passed and pr_pass

    # ---------------- 2. VITERBI CONVOLUTIONAL DECODER ----------------
    print("\n[STAGE 2] VITERBI CONVOLUTIONAL DECODER (K=7, Rate 1/2, 171/133 Octal):")
    conv_enc = ConvolutionalEncoder(constraint_length=7, polynomials=(0o171, 0o133))
    viterbi = ViterbiDecoder(constraint_length=7, polynomials=(0o171, 0o133))

    test_bits = rng.integers(0, 2, size=128, dtype=np.uint8)
    coded = conv_enc.encode(test_bits, terminate=True)
    res_clean = viterbi.decode_hard(coded, terminated=True)
    vit_clean_pass = np.array_equal(res_clean.decoded_bits, test_bits)
    print(f"  • Zero-Error Trellis Convergence:                {'[PASS]' if vit_clean_pass else '[FAIL]'}")
    all_passed = all_passed and vit_clean_pass

    # Noisy BER reduction
    noisy_coded = coded.copy()
    noisy_coded[10] ^= 1
    noisy_coded[45] ^= 1
    noisy_coded[90] ^= 1
    noisy_coded[150] ^= 1
    res_noisy = viterbi.decode_hard(noisy_coded, terminated=True)
    post_errs = int(np.sum(res_noisy.decoded_bits != test_bits))
    vit_noisy_pass = (post_errs == 0)
    print(f"  • Hard Decision Error Correction (4 bit flips):  {'[PASS]' if vit_noisy_pass else '[FAIL]'} (Residual Errors: {post_errs})")
    all_passed = all_passed and vit_noisy_pass

    # Soft Decision
    ideal_soft = np.where(coded == 0, 1.0, -1.0) + rng.normal(0, 0.35, size=len(coded))
    res_soft = viterbi.decode_soft(ideal_soft, terminated=True)
    soft_pass = np.array_equal(res_soft.decoded_bits, test_bits)
    print(f"  • Soft Decision (Signed LLR / AWGN):             {'[PASS]' if soft_pass else '[FAIL]'}")
    all_passed = all_passed and soft_pass

    # ---------------- 3. REED-SOLOMON RS(255, 223) GF(256) ----------------
    print("\n[STAGE 3] REED-SOLOMON ALGEBRAIC DECODER RS(255, 223) over GF(256):")
    rs_enc = ReedSolomonEncoder(n=255, k=223)
    rs_dec = ReedSolomonDecoder(n=255, k=223)

    rs_msg = rng.integers(0, 256, size=223, dtype=np.uint8).tobytes()
    rs_clean_cw = rs_enc.encode(rs_msg)

    # Test 0, 1, 4, 8, 16 errors
    rs_tiers = [0, 1, 4, 8, 16]
    rs_tier_pass = True
    for n_e in rs_tiers:
        cw_corrupt = bytearray(rs_clean_cw)
        if n_e > 0:
            pos = rng.choice(255, size=n_e, replace=False)
            for p in pos:
                cw_corrupt[p] ^= int(rng.integers(1, 256))
        res_rs = rs_dec.decode_block(bytes(cw_corrupt))
        if not (res_rs.success and res_rs.decoded_data == rs_msg and res_rs.corrected_symbol_count == n_e):
            rs_tier_pass = False
            print(f"  • RS({255},{223}) with {n_e} Symbol Errors:              [FAIL]")
            break
        else:
            print(f"  • RS({255},{223}) with {n_e} Symbol Errors:              [PASS] (Corrected {res_rs.corrected_symbol_count})")
    all_passed = all_passed and rs_tier_pass

    # Test 17 errors (exceeds t=16)
    cw_17 = bytearray(rs_clean_cw)
    pos_17 = rng.choice(255, size=17, replace=False)
    for p in pos_17:
        cw_17[p] ^= int(rng.integers(1, 256))
    res_17 = rs_dec.decode_block(bytes(cw_17))
    rs_17_pass = (res_17.uncorrectable or not res_rs.success)
    print(f"  • RS({255},{223}) with 17 Symbol Errors (> t=16):        {'[PASS]' if rs_17_pass else '[FAIL]'} (Correctly Flagged Uncorrectable)")
    all_passed = all_passed and rs_17_pass

    # ---------------- 4. FRAME SYNCHRONIZATION ----------------
    print("\n[STAGE 4] FRAME SYNCHRONIZATION & BOUNDARY EXTRACTION:")
    sync_word = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8) # 32-bit CCSDS
    stream = np.zeros(200, dtype=np.uint8)
    stream[50 : 50 + 32] = sync_word
    # Add 2 bit errors
    stream[52] ^= 1
    stream[60] ^= 1

    sync_res = SyncWordDetector.detect_sync_detailed(stream, sync_word, max_bit_errors=2)
    sync_pass = (sync_res.sync_detected and sync_res.first_sync_index == 50 and sync_res.min_hamming_distance == 2)
    print(f"  • Sliding Hamming Correlation (32-bit, 2 errors):{'[PASS]' if sync_pass else '[FAIL]'} (Confidence: {sync_res.best_confidence*100:.1f}%)")
    all_passed = all_passed and sync_pass

    # ---------------- 5. ENTROPY & PAYLOAD FORENSICS ----------------
    print("\n[STAGE 5] SHANNON ENTROPY & PAYLOAD CHARACTERIZATION:")
    presets = SyntheticForensicsDataset.generate_all_presets(length=223)
    forensics_pass = True
    for name, (p_data, exp_cat) in presets.items():
        char = PayloadEntropyAnalyzer.characterize_payload(p_data)
        if char.classification != exp_cat:
            forensics_pass = False
            print(f"  • Payload {name:<20}: [FAIL] (Got {char.classification}, expected {exp_cat})")
        else:
            print(f"  • Payload {name:<20}: [PASS] (H={char.entropy_bits_per_byte:.2f} b/B -> {char.classification})")
    all_passed = all_passed and forensics_pass

    # ---------------- 6. FULL CONCATENATED PIPELINE TEST ----------------
    print("\n[STAGE 6] FULL CONCATENATED FEC & FORENSICS PIPELINE TEST:")
    orig_payload = b"CRITICAL_TELEMETRY: SATELLITE_LINK_ACTIVE_PASS" + b" " * (223 - 46)
    cw = rs_enc.encode(orig_payload)
    cw_bits = np.unpackbits(np.frombuffer(cw, dtype=np.uint8))
    conv_bits = conv_enc.encode(cw_bits, terminate=True)
    
    # Pad to multiple of 64 for 8x8 block deinterleaver
    pad_needed = (64 - (len(conv_bits) % 64)) % 64
    active_bits = np.concatenate([conv_bits, np.zeros(pad_needed, dtype=np.uint8)])
    inter_bits = BlockInterleaver.interleave(active_bits, rows=8, cols=8)

    # Channel noise
    rx_bits = inter_bits.copy()
    rx_bits[40:46] ^= 1  # 6-bit burst error
    rx_bits[200] ^= 1

    # Receiver processing
    rx_deint = Deinterleaver.block_deinterleave(rx_bits, rows=8, cols=8)
    rx_vit = viterbi.decode_hard(rx_deint[: len(conv_bits)], terminated=True)
    rx_bytes = np.packbits(rx_vit.decoded_bits[: 255 * 8]).tobytes()
    rx_rs = rs_dec.decode_block(rx_bytes)

    e2e_pass = (rx_rs.success and rx_rs.decoded_data == orig_payload)
    print(f"  • End-to-End Concatenated Recovery:              {'[PASS]' if e2e_pass else '[FAIL]'}")
    all_passed = all_passed and e2e_pass

    print("\n" + "=" * 80)
    print(f"OVERALL VALIDATION RESULT: {'[PASS] ALL TESTS PASSED' if all_passed else '[FAIL] VALIDATION FAILED'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_fec_forensics_validation()
    sys.exit(0 if success else 1)
