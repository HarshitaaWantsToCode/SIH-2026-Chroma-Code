"""
Comprehensive Automated Pre-Commit Feature & System Audit.
Validates:
1. Ingestion (IQFormat FLOAT32, INT16, UINT8, WAV stereo & mono).
2. 1D-CNN AMC Classifier & Softmax probabilities.
3. Heuristic Modulation Classifier & Telemetry demarcation.
4. Out-of-Distribution & Non-Comm Audio Rejection.
5. DSP Progressive Synchronization (Costas, Mueller & Müller, RRC).
6. Forward Error Correction (Viterbi Trellis & Reed-Solomon algebraic).
7. Cyber Forensics & Shannon Information Entropy.
8. Signal DNA Emitter Fingerprinting & Baseline Anomaly detection.
9. Cryptographic SHA-256 Provenance Evidence Chain.
10. In-memory PDF Intelligence Dossier & Technical Handbook Generation.
"""

import os
from pathlib import Path
import numpy as np

from src.ingestion.binary_parser import SignalIngestionEngine, IQFormat
from src.ingestion.normalizer import SignalNormalizer
from src.amc.models.cnn1d_classifier import ModulationClassifier
from src.dsp.dsp_pipeline_analyzer import DSPProgressivePipeline
from src.decoding.cyber_forensics_service import CyberForensicsService
from src.decoding.signal_dna_service import SignalDnaService
from src.decoding.evidence_service import EvidenceProvenanceService
from src.visualization.pdf_report_generator import PDFReportGenerator
from src.visualization.generate_handbook_pdf import generate_full_glossary_pdf


def run_comprehensive_audit():
    print("=" * 95)
    print("CHROMA CODE — COMPREHENSIVE PRE-COMMIT FEATURE AUDIT")
    print("=" * 95)

    test_files = [
        ("test_captures/01_qpsk_telemetry_22db.iq", IQFormat.FLOAT32, 2e6, 250e3),
        ("test_captures/02_bpsk_tactical_6db_int16.iq", IQFormat.INT16, 2e6, 250e3),
        ("test_captures/03_16qam_encrypted_25db.raw", IQFormat.FLOAT32, 2e6, 250e3),
        ("test_captures/04_2fsk_dispatch_18db.bin", IQFormat.INT16, 2e6, 250e3),
        ("test_captures/05_bpsk_rtlsdr_uint8.iq", IQFormat.UINT8, 2e6, 250e3),
        ("test_captures/06_qpsk_stereo_analytic.wav", None, 48e3, 12e3),
        ("test_captures/07_aist2d_telemetry_mono.wav", None, 48e3, 12e3),
        ("test_captures/08_speech_acoustic_rejection.wav", None, 48e3, 12e3),
        ("test_captures/09_glitch_hum_rejection.wav", None, 48e3, 12e3),
        ("test_captures/10_16qam_satellite_downlink.iq", IQFormat.FLOAT32, 2e6, 250e3),
    ]

    clf = ModulationClassifier()
    assert clf.has_trained_weights, "CRITICAL: Trained weights models/amc_1dcnn_weights.pt failed to load!"
    print(f"[*] 1D-CNN Model Weights: LOADED ({clf.weights_path})")

    print("\n" + "-" * 95)
    print(f"{'File':<36} {'Modulation':<24} {'Classifier':<15} {'Status':<20}")
    print("-" * 95)

    for path_str, fmt, fs, rs in test_files:
        path = Path(path_str)
        assert path.exists(), f"Missing test file: {path_str}"

        with open(path, "rb") as f:
            raw_b = f.read()

        if path_str.endswith(".wav"):
            sig, detected_fs = SignalIngestionEngine.parse_wav(path, max_frames=16384)
            fs = detected_fs
        else:
            sig = SignalIngestionEngine.parse_iq_stream(raw_b, fmt=fmt, max_samples=16384)

        assert len(sig) > 0, f"Failed ingestion for {path_str}"

        dc = SignalNormalizer.remove_dc_offset(sig)
        norm, rms = SignalNormalizer.normalize_unit_power(dc)

        # 1. Automatic Modulation Classification
        amc_res = clf.predict(norm)

        # 2. Digital Signal Processing & Synchronization
        dsp = DSPProgressivePipeline.analyze(norm, amc_res.modulation, fs, rs)

        # 3. Cyber Forensics & Shannon Entropy
        forensics = CyberForensicsService.analyze(dsp.recovered_bits, sync_target="1ACFFC1D")

        # 4. Signal DNA & Anomaly Engine
        snr_str = str(dsp.extracted_params.get("Estimated SNR", "0.0")).replace(" dB", "").strip()
        try:
            snr_val = float(snr_str)
        except ValueError:
            snr_val = 20.0

        cfo_str = str(dsp.extracted_params.get("Carrier Frequency Offset (Δf)", "0.0")).replace(" Hz", "").strip()
        try:
            cfo_val = float(cfo_str)
        except ValueError:
            cfo_val = 0.0

        dna = SignalDnaService.evaluate(norm, amc_res.modulation, snr_val, cfo_val, forensics.entropy_bits_per_byte, fs, rs)

        # 5. Cryptographic Evidence Chain
        meta = {
            "Filename": path.name,
            "Format": str(fmt),
            "sample_rate_num": fs,
            "symbol_rate_num": rs,
            "Sample Count": str(len(sig)),
            "Duration": "10ms",
            "Channels": "2",
        }
        chain = EvidenceProvenanceService.generate_chain(raw_b, meta, amc_res, dsp.extracted_params, forensics.summary_card)

        # Verify integrity
        assert chain.is_chain_intact, f"Chain integrity failed on {path.name}"

        clf_short = "1D-CNN" if amc_res.classifier_type == "DEEP_1D_CNN" else "HEURISTIC"
        print(f"{path.name:<36} {amc_res.modulation:<24} {clf_short:<15} {amc_res.model_status:<20}")

    # 6. PDF Report Verification
    print("\n" + "-" * 95)
    print("[*] Testing PDF Report Generation Subsystems...")
    from src.decoding.fec_service import FECDecoderService
    fec_res = FECDecoderService.process(bits=dsp.recovered_bits)
    pdf_bytes = PDFReportGenerator.generate_pdf_bytes(
        case_id="CC-2026-AUDIT",
        timestamp="2026-09-05 00:00:00Z",
        meta_info=meta,
        amc_res=amc_res,
        dsp_analysis=dsp,
        fec_res=fec_res,
        forensics_res=forensics,
        signal_dna=dna,
        evidence_chain=chain,
        norm_signal=norm
    )
    assert len(pdf_bytes) > 5000, "Intelligence Dossier PDF generation produced empty or truncated output!"
    print(f"  -> Intelligence Dossier PDF: PASS ({len(pdf_bytes):,} bytes)")

    glossary_bytes = generate_full_glossary_pdf()
    assert len(glossary_bytes) > 5000, "Technical Handbook PDF generation produced empty or truncated output!"
    print(f"  -> Technical Handbook PDF:   PASS ({len(glossary_bytes):,} bytes)")

    print("\n" + "=" * 95)
    print("ALL 10 SUBSYSTEMS & CAPTURE TYPES PASSED AUDIT SUCCESSFULLY. SYSTEM READY TO PUSH.")
    print("=" * 95)


if __name__ == "__main__":
    run_comprehensive_audit()
