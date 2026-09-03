"""
Sync-Word Correlation, Frame Synchronization, and Boundary Extraction Module.

Identifies frame header markers using Hamming Distance cross-correlation
over demodulated binary bitstreams, measures confidence, and extracts structured frames.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class SyncMatch:
    """Individual synchronization word detection occurrence."""
    sync_start: int
    sync_end: int
    hamming_distance: int
    normalized_distance: float
    confidence: float


@dataclass
class ExtractedFrame:
    """Extracted frame bounded by synchronization markers."""
    frame_index: int
    frame_start: int
    frame_end: int
    frame_length: int
    sync_match: SyncMatch
    payload_bits: np.ndarray


@dataclass
class SyncDetectionResult:
    """Aggregated synchronization detection and framing result."""
    sync_detected: bool
    matches_count: int
    first_sync_index: Optional[int]
    min_hamming_distance: int
    best_confidence: float
    matches: List[SyncMatch]
    frames: List[ExtractedFrame]
    correlation_curve: np.ndarray


class SyncWordDetector:
    """
    Sliding window bit correlator with Hamming distance error thresholding
    and frame boundary segmentation.
    """

    @staticmethod
    def find_sync_word(
        bitstream: Union[np.ndarray, List[int]],
        sync_word: Union[np.ndarray, List[int]],
        max_bit_errors: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Locates sync header occurrences matching `sync_word` within `max_bit_errors`.
        Preserves backward compatibility with earlier project API.

        Mathematical Definition:
            d_H(u, v) = sum(u XOR v)

        Args:
            bitstream: 1D uint8 array containing demodulated bit sequence.
            sync_word: 1D uint8 pattern array (e.g., [1, 0, 1, 0, 1, 1, 0, 0]).
            max_bit_errors: Maximum allowable flipped bits (Hamming tolerance).

        Returns:
            List[Tuple[int, int]]: List of (start_index, hamming_distance).
        """
        res = SyncWordDetector.detect_sync_detailed(bitstream, sync_word, max_bit_errors)
        return [(m.sync_start, m.hamming_distance) for m in res.matches]

    @staticmethod
    def detect_sync_detailed(
        bitstream: Union[np.ndarray, List[int]],
        sync_word: Union[np.ndarray, List[int]],
        max_bit_errors: int = 4
    ) -> SyncDetectionResult:
        """
        Detailed sliding Hamming distance correlator computing confidence scores.

        Confidence Formulation:
            Let L = length of sync pattern.
            Hamming distance d_H in [0..L].
            Normalized distance = d_H / L
            Confidence = max(0.0, 1.0 - (d_H / L))
        """
        bits = np.asarray(bitstream, dtype=np.uint8)
        pat = np.asarray(sync_word, dtype=np.uint8)

        stream_len = len(bits)
        pat_len = len(pat)

        if pat_len == 0 or stream_len < pat_len:
            return SyncDetectionResult(
                sync_detected=False,
                matches_count=0,
                first_sync_index=None,
                min_hamming_distance=pat_len,
                best_confidence=0.0,
                matches=[],
                frames=[],
                correlation_curve=np.zeros(0, dtype=np.float32)
            )

        num_positions = stream_len - pat_len + 1
        corr_curve = np.zeros(num_positions, dtype=np.float32)
        matches: List[SyncMatch] = []
        min_hamming = pat_len

        for i in range(num_positions):
            window = bits[i : i + pat_len]
            # Bitwise XOR counts bit mismatches
            hamming_dist = int(np.sum(window ^ pat))
            norm_dist = hamming_dist / pat_len
            conf = max(0.0, 1.0 - norm_dist)
            corr_curve[i] = conf

            if hamming_dist < min_hamming:
                min_hamming = hamming_dist

            if hamming_dist <= max_bit_errors:
                matches.append(SyncMatch(
                    sync_start=i,
                    sync_end=i + pat_len,
                    hamming_distance=hamming_dist,
                    normalized_distance=norm_dist,
                    confidence=conf
                ))

        sync_detected = len(matches) > 0
        first_idx = matches[0].sync_start if sync_detected else None
        best_conf = matches[0].confidence if sync_detected else 0.0

        return SyncDetectionResult(
            sync_detected=sync_detected,
            matches_count=len(matches),
            first_sync_index=first_idx,
            min_hamming_distance=min_hamming,
            best_confidence=best_conf,
            matches=matches,
            frames=[],
            correlation_curve=corr_curve
        )

    @staticmethod
    def extract_frames(
        bitstream: Union[np.ndarray, List[int]],
        sync_word: Union[np.ndarray, List[int]],
        max_bit_errors: int = 4,
        frame_length: Optional[int] = None
    ) -> List[ExtractedFrame]:
        """
        Extracts structured frame boundaries based on detected sync words.
        
        Boundary Modes:
        1. Fixed Frame Length (when frame_length is provided):
           Frame spans [sync_start : sync_start + frame_length].
        2. Variable Frame Length (when frame_length is None):
           Frame spans from current sync_start up to the next detected sync_start (or end of stream).
        """
        bits = np.asarray(bitstream, dtype=np.uint8)
        det = SyncWordDetector.detect_sync_detailed(bits, sync_word, max_bit_errors)
        matches = det.matches
        frames: List[ExtractedFrame] = []

        if not matches:
            return frames

        pat_len = len(sync_word)
        total_len = len(bits)

        for idx, m in enumerate(matches):
            f_start = m.sync_start
            if frame_length is not None:
                f_end = min(total_len, f_start + frame_length)
            else:
                # Up to next sync match or end of stream
                if idx + 1 < len(matches):
                    f_end = matches[idx + 1].sync_start
                else:
                    f_end = total_len

            payload = bits[f_start + pat_len : f_end]
            frames.append(ExtractedFrame(
                frame_index=idx,
                frame_start=f_start,
                frame_end=f_end,
                frame_length=f_end - f_start,
                sync_match=m,
                payload_bits=payload
            ))

        return frames
