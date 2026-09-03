"""
Payload Information Entropy and Forensic Characterization Module.

Computes global and sliding-window Shannon Information Entropy, statistical profiling,
and conservative payload characterization (Plaintext-like, Structured/Binary, Compressed-like, High-Entropy).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class EntropyWindowStats:
    """Statistics for a single sliding window slice."""
    window_index: int
    byte_offset: int
    entropy: float
    unique_byte_count: int


@dataclass
class EntropyProfileResult:
    """Aggregate statistics and sliding-window entropy profile."""
    global_entropy: float
    window_size: int
    step_size: int
    windows: List[EntropyWindowStats]
    offsets: np.ndarray
    entropy_values: np.ndarray
    min_entropy: float
    max_entropy: float
    mean_entropy: float
    median_entropy: float
    std_entropy: float
    heatmap_matrix: np.ndarray


@dataclass
class PayloadCharacterizationResult:
    """Conservative forensic characterization of extracted byte payloads."""
    byte_count: int
    entropy_bits_per_byte: float
    printable_ascii_ratio: float
    null_byte_ratio: float
    unique_byte_ratio: float
    is_utf8_valid: bool
    repeated_pattern_indicator: bool
    classification: str
    interpretation: str
    confidence_description: str
    supporting_metrics: Dict[str, Union[float, int, bool, str]] = field(default_factory=dict)


class PayloadEntropyAnalyzer:
    """
    Computes Information-Theoretic metrics and statistical profiles for bitstreams and byte payloads.
    """

    @staticmethod
    def calculate_shannon_entropy(data: Union[np.ndarray, bytes, bytearray, List[int]]) -> float:
        """
        Calculates empirical Shannon Entropy H(X) in bits per byte (0.0 to 8.0):
            H(X) = - sum_{i=1}^M P(x_i) * log2( P(x_i) )

        Args:
            data: Binary byte array or uint8 numpy array.

        Returns:
            float: Shannon Entropy in bits/byte.
        """
        if isinstance(data, bytes) or isinstance(data, bytearray):
            arr = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, list):
            arr = np.array(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            if data.dtype == np.uint8 and (len(data) > 0 and np.all((data == 0) | (data == 1)) and len(data) >= 8):
                # If binary bits (0/1), pack into bytes
                arr = np.packbits(data)
            elif data.dtype != np.uint8:
                arr = np.packbits(data.astype(np.uint8))
            else:
                arr = data
        else:
            return 0.0

        if len(arr) == 0:
            return 0.0

        _, counts = np.unique(arr, return_counts=True)
        probabilities = counts / len(arr)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)

    @staticmethod
    def profile_sliding_window(
        data: Union[np.ndarray, bytes],
        window_size: int = 32,
        step_size: int = 8
    ) -> EntropyProfileResult:
        """
        Computes sliding-window entropy across the byte payload.

        Args:
            data: Raw bytes or packed uint8 array.
            window_size: Size of sliding window in bytes.
            step_size: Step size for window progression.
        """
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            if data.dtype != np.uint8 or (len(data) > 0 and np.all((data == 0) | (data == 1))):
                arr = np.packbits(data.astype(np.uint8))
            else:
                arr = data
        else:
            arr = np.array(data, dtype=np.uint8)

        global_h = PayloadEntropyAnalyzer.calculate_shannon_entropy(arr)

        if len(arr) < window_size:
            # Short payload: fallback to single global window
            single_win = EntropyWindowStats(
                window_index=0,
                byte_offset=0,
                entropy=global_h,
                unique_byte_count=len(np.unique(arr)) if len(arr) > 0 else 0
            )
            return EntropyProfileResult(
                global_entropy=global_h,
                window_size=window_size,
                step_size=step_size,
                windows=[single_win],
                offsets=np.array([0]),
                entropy_values=np.array([global_h]),
                min_entropy=global_h,
                max_entropy=global_h,
                mean_entropy=global_h,
                median_entropy=global_h,
                std_entropy=0.0,
                heatmap_matrix=np.array([[global_h]])
            )

        windows: List[EntropyWindowStats] = []
        offsets: List[int] = []
        h_vals: List[float] = []

        idx = 0
        for offset in range(0, len(arr) - window_size + 1, step_size):
            chunk = arr[offset : offset + window_size]
            _, counts = np.unique(chunk, return_counts=True)
            p = counts / len(chunk)
            h = float(-np.sum(p * np.log2(p)))
            u_count = len(counts)

            windows.append(EntropyWindowStats(
                window_index=idx,
                byte_offset=offset,
                entropy=h,
                unique_byte_count=u_count
            ))
            offsets.append(offset)
            h_vals.append(h)
            idx += 1

        h_arr = np.array(h_vals, dtype=np.float32)
        off_arr = np.array(offsets, dtype=np.int32)

        # Build 2D heatmap matrix (reshaped or repeated for visualization)
        heatmap_matrix = h_arr.reshape((1, -1))

        return EntropyProfileResult(
            global_entropy=global_h,
            window_size=window_size,
            step_size=step_size,
            windows=windows,
            offsets=off_arr,
            entropy_values=h_arr,
            min_entropy=float(np.min(h_arr)),
            max_entropy=float(np.max(h_arr)),
            mean_entropy=float(np.mean(h_arr)),
            median_entropy=float(np.median(h_arr)),
            std_entropy=float(np.std(h_arr)),
            heatmap_matrix=heatmap_matrix
        )

    @staticmethod
    def characterize_payload(data: Union[bytes, np.ndarray]) -> PayloadCharacterizationResult:
        """
        Performs conservative payload classification:
        - PLAINTEXT-LIKE (high printable ASCII / valid UTF-8, moderate entropy)
        - STRUCTURED/BINARY (low/moderate entropy, high nulls or repeating fields)
        - COMPRESSED-LIKE (entropy in 5.8 - 7.3 range, low printable ratio)
        - HIGH-ENTROPY (entropy > 7.35, uniform byte distribution)
        - UNKNOWN (insufficient data)
        """
        if isinstance(data, (bytes, bytearray)):
            byte_buf = bytes(data)
        elif isinstance(data, np.ndarray):
            if data.dtype == np.uint8 and np.all((data == 0) | (data == 1)):
                byte_buf = np.packbits(data).tobytes()
            else:
                byte_buf = data.tobytes()
        else:
            byte_buf = bytes(data)

        n_bytes = len(byte_buf)
        if n_bytes == 0:
            return PayloadCharacterizationResult(
                byte_count=0,
                entropy_bits_per_byte=0.0,
                printable_ascii_ratio=0.0,
                null_byte_ratio=0.0,
                unique_byte_ratio=0.0,
                is_utf8_valid=False,
                repeated_pattern_indicator=False,
                classification="UNKNOWN",
                interpretation="Empty payload buffer; no forensic assessment possible.",
                confidence_description="NO_DATA",
                supporting_metrics={}
            )

        arr = np.frombuffer(byte_buf, dtype=np.uint8)
        entropy = PayloadEntropyAnalyzer.calculate_shannon_entropy(arr)

        # 1. Printable ASCII Ratio (ASCII 32-126 plus newline \n, \r, \t)
        printable_mask = ((arr >= 32) & (arr <= 126)) | (arr == 9) | (arr == 10) | (arr == 13)
        printable_ratio = float(np.sum(printable_mask)) / n_bytes

        # 2. Null Byte Ratio (0x00)
        null_ratio = float(np.sum(arr == 0)) / n_bytes

        # 3. Unique Byte Ratio
        unique_bytes = len(np.unique(arr))
        unique_ratio = unique_bytes / min(256, n_bytes)

        # 4. UTF-8 Validity
        try:
            byte_buf.decode("utf-8")
            utf8_valid = True
        except UnicodeDecodeError:
            utf8_valid = False

        # 5. Repeated Pattern Indicator (check for exact period 2..16 repeats)
        repeated = False
        if n_bytes >= 16:
            for period in range(1, min(16, n_bytes // 4)):
                reps = n_bytes // period
                tile = np.tile(arr[:period], reps)
                if np.array_equal(arr[: len(tile)], tile):
                    repeated = True
                    break

        # Conservative Rule-Based Classification
        if printable_ratio >= 0.82 and entropy < 6.5:
            classification = "PLAINTEXT-LIKE"
            interpretation = "High printable ASCII density; consistent with human-readable text, commands, or unencrypted telemetry."
            conf_desc = f"High confidence ({printable_ratio*100:.1f}% printable ASCII)"
        elif null_ratio >= 0.3 or repeated or (entropy < 4.8 and unique_ratio < 0.4):
            classification = "STRUCTURED/BINARY"
            interpretation = "Low-to-moderate entropy with structured null fields or repetitive patterns; consistent with raw telemetry framing or binary sensor records."
            conf_desc = "Consistent structured framing"
        elif entropy >= 7.0 and unique_ratio > 0.5:
            classification = "HIGH-ENTROPY"
            interpretation = "High-entropy payload; characteristics may be consistent with compressed or encrypted data."
            conf_desc = f"Empirical Shannon entropy {entropy:.2f} bits/byte"
        elif entropy >= 5.0:
            classification = "COMPRESSED-LIKE"
            interpretation = "Elevated entropy density with non-uniform byte distribution; consistent with packed or compressed telemetry."
            conf_desc = f"Entropy {entropy:.2f} bits/byte"
        else:
            classification = "STRUCTURED/BINARY"
            interpretation = "Non-uniform intermediate entropy payload; characteristics consistent with binary record tables."
            conf_desc = f"Entropy {entropy:.2f} bits/byte"

        metrics = {
            "byte_count": n_bytes,
            "entropy": round(entropy, 4),
            "printable_ratio": round(printable_ratio, 4),
            "null_ratio": round(null_ratio, 4),
            "unique_bytes": unique_bytes,
            "unique_ratio": round(unique_ratio, 4),
            "utf8_valid": utf8_valid,
            "has_repetitive_pattern": repeated
        }

        return PayloadCharacterizationResult(
            byte_count=n_bytes,
            entropy_bits_per_byte=entropy,
            printable_ascii_ratio=printable_ratio,
            null_byte_ratio=null_ratio,
            unique_byte_ratio=unique_ratio,
            is_utf8_valid=utf8_valid,
            repeated_pattern_indicator=repeated,
            classification=classification,
            interpretation=interpretation,
            confidence_description=conf_desc,
            supporting_metrics=metrics
        )
