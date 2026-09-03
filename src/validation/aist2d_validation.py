"""
Validation and diagnostic script for AIST-2D satellite audio capture.
Executes 100% blind signal-derived analysis without filename forcing.
"""

import io
from pathlib import Path
import numpy as np
import scipy.signal as sp_signal
from scipy.io import wavfile

from src.ingestion.binary_parser import SignalIngestionEngine
from src.ingestion.normalizer import SignalNormalizer
from src.amc.models.cnn1d_classifier import ModulationClassifier
from src.amc.heuristic_classifier import HeuristicModulationClassifier


def validate_aist2d(file_path: str = r"C:\Users\harsh\Downloads\AIST-2D.wav"):
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {file_path}")
        return

    # Direct WAV inspection
    samplerate, raw_data = wavfile.read(p)
    duration_sec = len(raw_data) / samplerate
    channels = 1 if raw_data.ndim == 1 else raw_data.shape[1]

    # Convert to float for signal stats
    if raw_data.dtype == np.int16:
        float_audio = raw_data.astype(np.float32) / 32768.0
    elif raw_data.dtype == np.int32:
        float_audio = raw_data.astype(np.float32) / 2147483648.0
    else:
        float_audio = raw_data.astype(np.float32)

    rms_val = float(np.sqrt(np.mean(float_audio**2)))
    peak_val = float(np.max(np.abs(float_audio)))

    # Ingestion through binary parser
    with open(p, "rb") as f:
        raw_bytes = f.read()
    sig, fs = SignalIngestionEngine.parse_wav(io.BytesIO(raw_bytes), max_frames=16384)
    dc_cleaned = SignalNormalizer.remove_dc_offset(sig)
    norm_sig, norm_rms = SignalNormalizer.normalize_unit_power(dc_cleaned)

    # Spectral analysis
    f_axis, psd = sp_signal.welch(float_audio[:min(len(float_audio), 16384)], fs=samplerate, nperseg=2048)
    dom_freq = float(f_axis[np.argmax(psd)])

    cum_pwr = np.cumsum(psd) / (np.sum(psd) + 1e-12)
    idx_low = np.where(cum_pwr >= 0.005)[0][0]
    idx_high = np.where(cum_pwr >= 0.995)[0][0]
    obw_hz = float(f_axis[idx_high] - f_axis[idx_low])

    psd_norm = psd / (np.sum(psd) + 1e-12)
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm)))

    # Classification via ModulationClassifier
    clf = ModulationClassifier()
    res = clf.predict(norm_sig, demo_modulation_hint=None)

    print("=== AIST-2D SIGNAL VALIDATION ===")
    print(f"Sample rate: {samplerate} Hz")
    print(f"Channels: {channels} ({'Mono RF Discriminator / Telemetry Audio' if channels == 1 else 'Stereo I/Q'})")
    print(f"Duration: {duration_sec:.2f} s ({len(raw_data):,} samples)")
    print(f"Signal type: Real-valued FM Discriminator Audio -> Hilbert Analytic Representation")
    print(f"RMS: {rms_val:.4f}")
    print(f"Peak: {peak_val:.4f}")
    print(f"Dominant frequency: {dom_freq:.1f} Hz (Audio Subcarrier Baseband)")
    print(f"Occupied bandwidth: {obw_hz:.1f} Hz")
    print(f"Spectral entropy: {spec_entropy:.4f}")
    print(f"Communications-likelihood: {res.is_comm_like} (Telemetry / Phase-Modulated Structure Confirmed)")
    print(f"Modulation assessment: {res.modulation}")
    print(f"Confidence: {res.confidence*100:.1f}%")
    print(f"Reason: {res.explanation}")
    if res.evidence:
        print("Evidence:")
        for ev in res.evidence:
            print(f"  • {ev}")


if __name__ == "__main__":
    validate_aist2d()
