"""
Signal Conditioning and Normalization Module.

Provides mathematical operators for zero-centering (DC offset mitigation) and
variance scaling (Unit Energy / Constant False-Alarm Rate compliance).
"""

from typing import Tuple
import numpy as np


class SignalNormalizer:
    """
    Applies deterministic statistical normalization over raw complex baseband signals.
    """

    @staticmethod
    def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
        """
        Removes baseband DC bias (carrier feedthrough / ADC offset):
            s_clean[n] = s[n] - E[s[n]]

        Args:
            signal: 1D complex NumPy array.

        Returns:
            np.ndarray: DC-removed analytic signal.
        """
        dc_bias = np.mean(signal)
        return signal - dc_bias

    @staticmethod
    def normalize_unit_power(signal: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
        """
        Scales signal to have average unit power (RMS = 1.0):
            P_avg = (1 / N) * sum(|s[n]|^2)
            s_norm[n] = s[n] / sqrt(P_avg + eps)

        Args:
            signal: 1D complex NumPy array.
            eps: Epsilon floor to prevent division by zero in noise-only buffers.

        Returns:
            Tuple[np.ndarray, float]: (Normalized signal, Root Mean Square scaling factor).
        """
        pwr = np.mean(np.abs(signal) ** 2)
        if not np.isfinite(pwr) or pwr < eps:
            rms_power = 1.0
            normalized_signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            rms_power = float(np.sqrt(pwr))
            normalized_signal = signal / rms_power
            normalized_signal = np.nan_to_num(normalized_signal, nan=0.0, posinf=1.0, neginf=-1.0)

        return normalized_signal, float(rms_power)

    @staticmethod
    def normalize_peak(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Scales signal so max envelope amplitude equals 1.0:
            s_peak[n] = s[n] / (max(|s[n]|) + eps)
        """
        peak_amp = np.max(np.abs(signal)) + eps
        return signal / peak_amp
