"""
Deterministic Synthetic Forensics Payloads Generator for Test Vectors and Validation.

Generates ground-truth datasets for:
1. Plaintext ASCII Telemetry
2. Structured Binary Frame Records
3. Compressed-like Payload
4. High-Entropy Pseudorandom Stream
5. Encrypted-like Ciphertext Payload
"""

import struct
import zlib
from typing import Dict, Tuple
import numpy as np


class SyntheticForensicsDataset:
    """
    Deterministic dataset generator producing ground-truth test vectors for forensics.
    """

    @staticmethod
    def generate_plaintext_payload(length: int = 223, seed: int = 42) -> bytes:
        """
        Generates readable ASCII telemetry stream.
        Expected entropy: ~4.0 - 5.2 bits/byte, >95% printable ASCII.
        """
        rng = np.random.default_rng(seed)
        sentences = [
            b"TELEMETRY_PACKET: STATUS=NOMINAL, VOLTAGE=28.4V, CURRENT=1.2A, TEMP=+24.1C. ",
            b"ORBIT_EPHEMERIS: ALT=540.2KM, VELOCITY=7.58KM/S, INCLINATION=97.4DEG. ",
            b"HEALTH_CHECK: SENSOR_ARRAY_A=OK, GYROSCOPE=LOCKED, LINK_MARGIN=14.2DB. ",
            b"SUBSYSTEM_PING: PAYLOAD_INSTRUMENT=ACTIVE, BUFFER_UTILIZATION=34.2%. "
        ]
        out = bytearray()
        while len(out) < length:
            idx = int(rng.integers(0, len(sentences)))
            out.extend(sentences[idx])
        return bytes(out[:length])

    @staticmethod
    def generate_structured_binary_payload(length: int = 223, seed: int = 42) -> bytes:
        """
        Generates binary telemetry records (headers, counters, timestamps, null padding).
        Expected entropy: ~2.5 - 4.5 bits/byte, high null/pattern ratio.
        """
        rng = np.random.default_rng(seed)
        out = bytearray()
        seq_num = 1000
        while len(out) < length:
            # Struct: 4-byte Magic (0xDEADBEEF), 4-byte uint32 counter, 4-byte float, 4 bytes zeros
            magic = 0xDEADBEEF
            val = float(rng.uniform(10.0, 50.0))
            record = struct.pack(">II f 4x", magic, seq_num, val)
            out.extend(record)
            seq_num += 1
        return bytes(out[:length])

    @staticmethod
    def generate_compressed_payload(length: int = 223, seed: int = 42) -> bytes:
        """
        Generates zlib-compressed payload from structured source.
        Expected entropy: ~6.0 - 7.3 bits/byte.
        """
        # Compress realistic structured sentences to generate realistic compression byte distribution
        source = (
            b"MISSION_CONTROL_TELEMETRY_LOG_FRAME_START: "
            b"PRIMARY_TRANSPONDER_FREQUENCY=2.245GHZ, AZIMUTH=142.5DEG, ELEVATION=45.1DEG, "
            b"DOWNLINK_POWER_DBM=43.2, SOLAR_PANEL_EFFICIENCY=98.1%, ONBOARD_CLOCK_DRIFT=0.002MS. "
        ) * 20
        compressed = zlib.compress(source, level=6)
        if len(compressed) >= length:
            return compressed[:length]
        pad = (compressed * ((length // len(compressed)) + 1))[:length]
        return pad

    @staticmethod
    def generate_high_entropy_payload(length: int = 223, seed: int = 42) -> bytes:
        """
        Generates uniform pseudorandom bytes.
        Expected entropy: ~7.80 - 7.99 bits/byte.
        """
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 256, size=length, dtype=np.uint8)
        return raw.tobytes()

    @staticmethod
    def generate_all_presets(length: int = 223) -> Dict[str, Tuple[bytes, str]]:
        """Returns dictionary of all standard test payloads with expected classifications."""
        return {
            "PLAINTEXT_ASCII": (
                SyntheticForensicsDataset.generate_plaintext_payload(length, seed=42),
                "PLAINTEXT-LIKE"
            ),
            "STRUCTURED_BINARY": (
                SyntheticForensicsDataset.generate_structured_binary_payload(length, seed=42),
                "STRUCTURED/BINARY"
            ),
            "COMPRESSED_DATA": (
                SyntheticForensicsDataset.generate_compressed_payload(length, seed=42),
                "COMPRESSED-LIKE"
            ),
            "HIGH_ENTROPY_RANDOM": (
                SyntheticForensicsDataset.generate_high_entropy_payload(length, seed=42),
                "HIGH-ENTROPY"
            )
        }
