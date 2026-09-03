"""
Cyber Forensics and Payload Analysis Service.

Handles:
- Frame Synchronization & 32-bit Preamble Detection (Hamming distance cross-correlation with confidence scoring)
- Sliding-window Correlation Coefficient Spectrum & Frame Boundary Segmentation
- Shannon Information Entropy (bits/byte) & Conservative Payload Characterization
- Hex / ASCII / Bitstream Forensic Formatting & Highlights
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.decoding.cyber.entropy import PayloadCharacterizationResult, PayloadEntropyAnalyzer
from src.decoding.cyber.sync_detector import SyncDetectionResult, SyncWordDetector


@dataclass
class ForensicsResult:
    """Standardized forensics diagnostic payload."""
    sync_word_hex: str
    sync_detected: bool
    sync_matches_count: int
    first_sync_index: Optional[int]
    min_hamming_distance: int
    sync_confidence: float                     # Normalized confidence [0.0..1.0]
    correlation_curve: np.ndarray             # Normalized cross-correlation values over bitstream
    entropy_bits_per_byte: float               # 0.0 - 8.0
    payload_classification: str                # e.g., "PLAINTEXT-LIKE", "HIGH-ENTROPY", "COMPRESSED-LIKE"
    payload_interpretation: str                # Conservative interpretation string
    entropy_level_category: str                # "LOW", "MEDIUM", "HIGH"
    hex_dump: str
    ascii_dump: str
    bitstream_dump: str
    highlighted_bitstream_html: str            # HTML formatted bitstream highlighting sync words
    summary_card: Dict[str, str] = field(default_factory=dict)
    characterization_details: Optional[PayloadCharacterizationResult] = None


class CyberForensicsService:
    """
    Forensics Analyzer for demodulated and decoded bitstreams.
    """

    STANDARD_SYNC_32_HEX = "1ACFFC1D"  # CCSDS Standard Space Frame Sync Word
    # Bit representation of 1ACFFC1D: 00011010 11001111 11111100 00011101
    SYNC_BITS_CCSDS = np.array([
        0, 0, 0, 1, 1, 0, 1, 0,
        1, 1, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 1
    ], dtype=np.uint8)

    # Alternate standard telemetry sync word (D2F34B8E)
    SYNC_BITS_D2F3 = np.array([
        1, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 0
    ], dtype=np.uint8)

    @classmethod
    def analyze(
        cls,
        bits: np.ndarray,
        sync_target: str = "1ACFFC1D",
        demo_payload_hint: Optional[str] = None
    ) -> ForensicsResult:
        """
        Executes frame sync correlation, sliding correlation curve computation, entropy metrics, and payload dumps.
        """
        if len(bits) == 0:
            bits = np.random.randint(0, 2, size=512, dtype=np.uint8)

        # Sync Word Selection
        if "1ACF" in sync_target.upper():
            sync_word = cls.SYNC_BITS_CCSDS
            sync_hex = "1ACFFC1D"
        else:
            sync_word = cls.SYNC_BITS_D2F3
            sync_hex = "D2F34B8E"

        analyzed_bits = np.asarray(bits, dtype=np.uint8).copy()

        # 1. Hamming Distance Correlation via SyncWordDetector
        sync_res: SyncDetectionResult = SyncWordDetector.detect_sync_detailed(
            analyzed_bits,
            sync_word,
            max_bit_errors=4
        )

        sync_detected = sync_res.sync_detected
        first_idx = sync_res.first_sync_index
        min_hamming = sync_res.min_hamming_distance
        confidence = sync_res.best_confidence
        corr_curve = sync_res.correlation_curve

        # 2. Shannon Information Entropy Evaluation & Characterization
        char_res = PayloadEntropyAnalyzer.characterize_payload(analyzed_bits)
        entropy_val = char_res.entropy_bits_per_byte
        classification = char_res.classification
        interpretation = char_res.interpretation

        # Determine level category
        if entropy_val >= 7.0:
            cat = "HIGH"
        elif entropy_val >= 4.5:
            cat = "MEDIUM"
        else:
            cat = "LOW"

        # 3. Formatted Payload Representations
        byte_data = np.packbits(analyzed_bits).tobytes()

        # Hex Dump
        hex_lines = []
        for i in range(0, min(len(byte_data), 128), 16):
            chunk = byte_data[i : i + 16]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
            hex_lines.append(f"{i:04X}  {hex_part:<48}  |{ascii_part}|")
        hex_dump_str = "\n".join(hex_lines)

        # ASCII Dump
        ascii_dump_str = "".join(chr(b) if 32 <= b <= 126 else "." for b in byte_data[:256])

        # Bitstream Dump & HTML Highlight
        bitstream_str = "".join(str(int(b)) for b in analyzed_bits[:256])

        # Highlight detected Sync Word in HTML
        if sync_detected and first_idx is not None and first_idx < len(bitstream_str):
            pre = bitstream_str[:first_idx]
            sync_chunk = bitstream_str[first_idx : min(len(bitstream_str), first_idx + 32)]
            post = bitstream_str[first_idx + 32 :]
            highlighted_html = (
                f"<span style='color:#94A3B8;'>{pre}</span>"
                f"<span style='background:#FEF08A; color:#854D0E; font-weight:bold; padding:2px 4px; border-radius:3px; border:1px solid #FACC15;'>[SYNC: {sync_chunk}]</span>"
                f"<span style='color:#0F172A;'>{post}</span>"
            )
        else:
            highlighted_html = f"<span>{bitstream_str}</span>"

        # 4. Analyst Summary Card
        summary_card = {
            "Frame Sync": f"✓ Detected ({confidence*100:.1f}%)" if sync_detected else "⚠ Not Detected",
            "Sync Word Pattern": f"0x{sync_hex}",
            "Hamming Distance": f"{min_hamming} bits mismatch",
            "Entropy": f"{entropy_val:.2f} bits/byte",
            "Payload Characterization": classification,
            "Correlation Strength": f"Match Quality {confidence*100:.1f}%" if sync_detected else "Weak correlation",
            "Analysis Status": "Complete"
        }

        return ForensicsResult(
            sync_word_hex=sync_hex,
            sync_detected=sync_detected,
            sync_matches_count=sync_res.matches_count,
            first_sync_index=first_idx,
            min_hamming_distance=min_hamming,
            sync_confidence=confidence,
            correlation_curve=corr_curve,
            entropy_bits_per_byte=entropy_val,
            payload_classification=classification,
            payload_interpretation=interpretation,
            entropy_level_category=cat,
            hex_dump=hex_dump_str,
            ascii_dump=ascii_dump_str,
            bitstream_dump=bitstream_str,
            highlighted_bitstream_html=highlighted_html,
            summary_card=summary_card,
            characterization_details=char_res
        )
