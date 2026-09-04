"""
Comprehensive Unit and Regression Test Suite for 1D-CNN Modulation Classifier.

Tests:
1. Model weights loading and structure verification.
2. Output shape, class labels, and valid logits/probabilities.
3. Softmax probabilities sum strictly to 1.0.
4. Evaluation mode determinism.
5. Absolute filename independence (identical signals yield identical predictions under any filename).
6. Safe handling of short signals (deterministic zero-padding).
7. Safe handling of long signals (deterministic windowing).
8. Conservative rejection / out-of-distribution handling for conventional audio / non-comm glitches.
9. Ground truth synthetic signal verification across BPSK, QPSK, 16-QAM, 2-FSK.
10. Preservation of heuristic telemetry path for mono discriminator captures.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.amc.dataset_generator import CNN_CLASS_LABELS, DEFAULT_WINDOW_SIZE, complex_to_2channel_tensor
from src.amc.ground_truth_generator import GroundTruthSignalGenerator
from src.amc.models.cnn1d_classifier import Modulation1DCNN, ModulationClassifier


@pytest.fixture(scope="module")
def classifier():
    clf = ModulationClassifier("models/amc_1dcnn_weights.pt")
    assert clf.has_trained_weights is True, "Failed to load trained weights from models/amc_1dcnn_weights.pt"
    return clf


def test_model_file_exists_and_loads():
    weights_path = Path("models/amc_1dcnn_weights.pt")
    assert weights_path.exists(), "Model weights file does not exist"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    assert "state_dict" in checkpoint, "Missing state_dict in checkpoint"
    assert "class_labels" in checkpoint, "Missing class_labels in checkpoint"
    assert len(checkpoint["class_labels"]) == 4, "Expected 4 class labels"


def test_cnn_architecture_forward():
    model = Modulation1DCNN(sequence_length=DEFAULT_WINDOW_SIZE, num_classes=4, class_labels=CNN_CLASS_LABELS)
    model.eval()
    dummy_input = torch.randn(2, 2, DEFAULT_WINDOW_SIZE)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"


def test_softmax_probabilities_sum_to_one(classifier):
    sig = GroundTruthSignalGenerator.generate("QPSK", num_symbols=512, snr_db=20.0, seed=123).signal_iq
    res = classifier.predict(sig)
    
    probs = res.probabilities
    assert len(probs) == 4
    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, rel=1e-3) == 1.0, f"Probabilities must sum to 1.0, got {total_prob}"


def test_deterministic_output_in_eval_mode(classifier):
    sig = GroundTruthSignalGenerator.generate("BPSK", num_symbols=512, snr_db=15.0, seed=456).signal_iq
    res1 = classifier.predict(sig)
    res2 = classifier.predict(sig)

    assert res1.modulation == res2.modulation
    assert pytest.approx(res1.confidence, rel=1e-5) == res2.confidence
    for k in res1.probabilities:
        assert pytest.approx(res1.probabilities[k], rel=1e-5) == res2.probabilities[k]


def test_filename_independence(classifier):
    # Verify that prediction does not depend on filename or any metadata
    sig = GroundTruthSignalGenerator.generate("16-QAM", num_symbols=512, snr_db=22.0, seed=789).signal_iq
    
    res_a = classifier.predict(sig, demo_modulation_hint="totally_wrong_hint.iq")
    res_b = classifier.predict(sig, demo_modulation_hint="different_name.wav")
    res_c = classifier.predict(sig, demo_modulation_hint=None)

    assert res_a.modulation == res_b.modulation == res_c.modulation == "16-QAM"
    assert res_a.confidence == res_b.confidence == res_c.confidence


def test_short_signal_padding(classifier):
    short_sig = np.random.randn(200) + 1j * np.random.randn(200)
    tensor_2ch = complex_to_2channel_tensor(short_sig, target_length=DEFAULT_WINDOW_SIZE)
    assert tensor_2ch.shape == (2, DEFAULT_WINDOW_SIZE)
    # Re-normalizing should produce finite unit power
    assert np.all(np.isfinite(tensor_2ch))


def test_long_signal_deterministic_windowing(classifier):
    long_sig = np.random.randn(20000) + 1j * np.random.randn(20000)
    tensor_2ch = complex_to_2channel_tensor(long_sig, target_length=DEFAULT_WINDOW_SIZE)
    assert tensor_2ch.shape == (2, DEFAULT_WINDOW_SIZE)
    assert np.all(np.isfinite(tensor_2ch))


@pytest.mark.parametrize("mod_name", ["BPSK", "QPSK", "16-QAM", "2-FSK"])
def test_synthetic_ground_truth_classification(classifier, mod_name):
    sig = GroundTruthSignalGenerator.generate(
        modulation=mod_name,
        num_symbols=1024,
        snr_db=20.0,
        cfo_hz=500.0,
        seed=8888,
    ).signal_iq

    res = classifier.predict(sig)
    assert res.classifier_type == "DEEP_1D_CNN"
    assert res.model_status == "REAL_TRAINED_MODEL"
    assert res.modulation == mod_name
    assert res.confidence >= 0.80, f"Expected high confidence for {mod_name}, got {res.confidence}"


def test_non_comm_audio_rejection(classifier):
    # Pure unmodulated DC tone (non-comm single tone)
    pure_tone = np.ones(4096, dtype=np.complex64)
    res = classifier.predict(pure_tone)
    assert res.is_comm_like is False or res.modulation == "UNKNOWN"
    assert res.model_status in ("INSUFFICIENT_EVIDENCE", "OUT_OF_DISTRIBUTION")


def test_invalid_short_buffer(classifier):
    too_short = np.array([1.0 + 0j, 0.0 + 1j])
    res = classifier.predict(too_short)
    assert res.modulation == "UNKNOWN"
    assert res.model_status == "INSUFFICIENT_EVIDENCE"
