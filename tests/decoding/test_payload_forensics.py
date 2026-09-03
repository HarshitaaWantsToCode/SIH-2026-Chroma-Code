"""
Tests for Payload Forensics and Conservative Characterization.
"""

import numpy as np
import pytest
from src.decoding.cyber.entropy import PayloadEntropyAnalyzer
from src.decoding.cyber.forensics_payloads import SyntheticForensicsDataset


def test_payload_characterization_categories():
    """Validates conservative classification of synthetic telemetry, binary, compressed, and random datasets."""
    presets = SyntheticForensicsDataset.generate_all_presets(length=256)

    for name, (data, expected_cat) in presets.items():
        res = PayloadEntropyAnalyzer.characterize_payload(data)
        assert res.classification == expected_cat, f"Mismatch for preset {name}: got {res.classification}, expected {expected_cat}"


def test_payload_empty_and_short():
    """Validates safe handling of empty and short byte streams."""
    empty_res = PayloadEntropyAnalyzer.characterize_payload(b"")
    assert empty_res.classification == "UNKNOWN"
    assert empty_res.byte_count == 0
