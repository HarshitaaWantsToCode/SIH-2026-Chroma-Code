"""
Forensic Report Object Definitions.

Encapsulates complete provenance, framing, payload characterization, and FEC recovery evidence.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import numpy as np


@dataclass
class InputMetadataReport:
    """Metadata regarding raw ingested capture."""
    filename: str
    format: str
    sample_count: int
    sample_rate_hz: float
    duration_sec: float


@dataclass
class FrameSyncReport:
    """Synchronization marker and frame extraction evidence."""
    sync_word_hex: str
    sync_confidence: float
    min_hamming_distance: int
    frame_start_bit: Optional[int]
    frame_length_bits: Optional[int]
    detected: bool


@dataclass
class PayloadForensicReport:
    """Information entropy and payload characterization metrics."""
    byte_count: int
    shannon_entropy: float
    printable_ascii_ratio: float
    null_byte_ratio: float
    classification: str
    interpretation: str
    confidence_description: str


@dataclass
class FECRecoveryReport:
    """Forward Error Correction recovery statistics."""
    deinterleaver_type: str
    viterbi_status: str
    viterbi_bit_errors_estimated: int
    reed_solomon_status: str
    reed_solomon_symbols_corrected: int
    reed_solomon_uncorrectable: bool
    overall_recovery_success: bool


@dataclass
class ComprehensiveForensicReport:
    """Master multi-layer forensic report matching Phase 11 specification."""
    input_metadata: InputMetadataReport
    frame_sync: FrameSyncReport
    payload_forensics: PayloadForensicReport
    fec_recovery: FECRecoveryReport
    sha256_evidence_identifier: str
    hex_preview: str
    ascii_preview: str
    summary_card: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def compute_evidence_hash(data: bytes) -> str:
        """Computes verifiable SHA-256 evidence identifier."""
        return hashlib.sha256(data).hexdigest()
