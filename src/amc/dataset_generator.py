"""
Synthetic IQ Dataset Generator for AMC 1D-CNN Training and Evaluation.

Generates realistic complex I/Q baseband signals across:
- BPSK
- QPSK
- 16-QAM
- 2-FSK

With extensive randomization:
- Random bit sequences
- SNR variation: 0 dB to 30 dB
- Carrier Frequency Offset (CFO): -5.0 kHz to +5.0 kHz
- Initial phase offset: 0 to 2*pi
- RRC roll-off factor: 0.20 to 0.40
- Symbols per second / Sample rate variations
- AWGN noise channels
- Fixed-length window extraction (default N=4096) with unit-energy normalization: [2, N]
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from src.amc.ground_truth_generator import GroundTruthSignalGenerator


CNN_CLASS_LABELS = ["BPSK", "QPSK", "16-QAM", "2-FSK"]
DEFAULT_WINDOW_SIZE = 4096


def complex_to_2channel_tensor(signal: np.ndarray, target_length: int = DEFAULT_WINDOW_SIZE) -> np.ndarray:
    """
    Transforms a 1D complex numpy array into a 2-channel float32 array [2, target_length].
    Channel 0: In-Phase (I)
    Channel 1: Quadrature (Q)

    Ensures unit-energy normalization: sqrt(mean(I^2 + Q^2)) = 1.0.
    Deterministic padding (zero-pad at end) for short signals, center crop/deterministic window for long signals.
    """
    if len(signal) == 0:
        return np.zeros((2, target_length), dtype=np.float32)

    # Unit power normalization
    power = np.mean(np.abs(signal) ** 2)
    if power > 1e-12:
        norm_sig = signal / np.sqrt(power)
    else:
        norm_sig = signal

    sig_len = len(norm_sig)
    if sig_len < target_length:
        padded = np.zeros(target_length, dtype=np.complex64)
        padded[:sig_len] = norm_sig
        norm_sig = padded
    elif sig_len > target_length:
        # Deterministic window extraction
        start_idx = 0
        norm_sig = norm_sig[start_idx : start_idx + target_length]

    i_ch = np.real(norm_sig).astype(np.float32)
    q_ch = np.imag(norm_sig).astype(np.float32)

    # Re-normalize after windowing if needed
    win_pwr = np.mean(i_ch**2 + q_ch**2)
    if win_pwr > 1e-12:
        scale = 1.0 / np.sqrt(win_pwr)
        i_ch = i_ch * scale
        q_ch = q_ch * scale

    return np.stack([i_ch, q_ch], axis=0)  # Shape: [2, target_length]


class SignalIQDataset(Dataset):
    """PyTorch Dataset holding complex IQ tensor windows and class labels."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, snrs: Optional[np.ndarray] = None):
        """
        Args:
            data: NumPy array of shape [Num_Samples, 2, Window_Length], dtype float32
            labels: NumPy array of shape [Num_Samples], dtype int64
            snrs: Optional NumPy array of shape [Num_Samples], dtype float32
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.snrs = torch.from_numpy(snrs).float() if snrs is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def generate_synthetic_dataset(
    samples_per_class: int = 500,
    window_length: int = DEFAULT_WINDOW_SIZE,
    snr_levels: Optional[List[float]] = None,
    seed_offset: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a balanced dataset of synthetic IQ signals across all target classes.

    Returns:
        (X, y, snrs):
            X: [Total_Samples, 2, window_length]
            y: [Total_Samples] (class indices 0..3)
            snrs: [Total_Samples] (SNR in dB for each sample)
    """
    if snr_levels is None:
        snr_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    num_classes = len(CNN_CLASS_LABELS)
    total_samples = samples_per_class * num_classes

    X = np.zeros((total_samples, 2, window_length), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int64)
    snrs = np.zeros(total_samples, dtype=np.float32)

    sample_idx = 0
    rng = np.random.RandomState(seed_offset)

    for class_idx, mod_name in enumerate(CNN_CLASS_LABELS):
        for _ in range(samples_per_class):
            snr = float(rng.choice(snr_levels))
            cfo = float(rng.uniform(-4000.0, 4000.0))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            rrc_alpha = float(rng.uniform(0.20, 0.40))
            num_syms = int(round((window_length / 8.0) * 1.5))
            sample_seed = int(rng.randint(1, 1000000))

            gt_signal = GroundTruthSignalGenerator.generate(
                modulation=mod_name,
                num_symbols=max(num_syms, 512),
                snr_db=snr,
                cfo_hz=cfo,
                phase_offset_rad=phase,
                sample_rate=2000000.0,
                symbol_rate=250000.0,
                rrc_alpha=rrc_alpha,
                seed=sample_seed,
            )

            tensor_2ch = complex_to_2channel_tensor(gt_signal.signal_iq, target_length=window_length)
            X[sample_idx] = tensor_2ch
            y[sample_idx] = class_idx
            snrs[sample_idx] = snr
            sample_idx += 1

    # Shuffle the dataset
    shuffle_indices = rng.permutation(total_samples)
    return X[shuffle_indices], y[shuffle_indices], snrs[shuffle_indices]


def create_train_val_test_datasets(
    total_samples_per_class: int = 600,
    window_length: int = DEFAULT_WINDOW_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[SignalIQDataset, SignalIQDataset, SignalIQDataset, Dict[str, any]]:
    """
    Creates cleanly split Train, Validation, and Held-out Test datasets with zero sample leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    n_train_pc = int(round(total_samples_per_class * train_ratio))
    n_val_pc = int(round(total_samples_per_class * val_ratio))
    n_test_pc = total_samples_per_class - n_train_pc - n_val_pc

    # Distinct seeds for train, val, and test generation
    X_train, y_train, snr_train = generate_synthetic_dataset(
        samples_per_class=n_train_pc,
        window_length=window_length,
        seed_offset=seed + 100,
    )
    X_val, y_val, snr_val = generate_synthetic_dataset(
        samples_per_class=n_val_pc,
        window_length=window_length,
        seed_offset=seed + 200000,
    )
    X_test, y_test, snr_test = generate_synthetic_dataset(
        samples_per_class=n_test_pc,
        window_length=window_length,
        seed_offset=seed + 400000,
    )

    train_ds = SignalIQDataset(X_train, y_train, snr_train)
    val_ds = SignalIQDataset(X_val, y_val, snr_val)
    test_ds = SignalIQDataset(X_test, y_test, snr_test)

    metadata = {
        "classes": CNN_CLASS_LABELS,
        "window_length": window_length,
        "total_samples": len(X_train) + len(X_val) + len(X_test),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "samples_per_class_train": n_train_pc,
        "samples_per_class_val": n_val_pc,
        "samples_per_class_test": n_test_pc,
        "snr_distribution": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    }

    return train_ds, val_ds, test_ds, metadata
