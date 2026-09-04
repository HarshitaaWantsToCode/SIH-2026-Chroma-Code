"""
Symbol Timing Synchronization: Mueller and Müller Algorithm.

Decision-directed timing error detector (TED) operating at 1 or 2 samples per symbol
to resolve fractional delay and sample strobe instances without explicit interpolation filters.
"""

from typing import Tuple
import numpy as np


class MuellerMullerTimingRecovery:
    """
    Interpolated Mueller & Müller Discrete-Time Clock Recovery.
    """

    def __init__(
        self,
        samples_per_symbol: int,
        gain: float = 0.01,
        damping: float = 0.707
    ) -> None:
        """
        Args:
            samples_per_symbol: Nominal oversampling ratio (Fs / Rs).
            gain: Tracking loop gain parameter.
            damping: Loop damping factor.
        """
        self.sps = samples_per_symbol
        self.gain = gain
        self.damping = damping

    def recover_symbols(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts synchronized symbol-spaced samples from oversampled complex input.

        Timing Error Detector (Decision-Directed M&M):
            e[k] = Re{ x[k] * conj(a[k-1]) - x[k-1] * conj(a[k]) }
            where x[k] is the received sample and a[k] is the quantized decision symbol.

        Args:
            signal: Matched-filtered complex input array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Synchronized symbols array, Extracted strobe indices).
        """
        num_samples = len(signal)
        symbols = []
        strobe_indices = []

        idx = float(self.sps)
        fractional_offset = 0.0
        omega = float(self.sps)  # Instantaneous step interval

        prev_sample = 0.0 + 0.0j
        prev_decision = 0.0 + 0.0j

        while np.isfinite(idx) and int(idx) < num_samples - 1:
            base_idx = int(idx)
            mu = idx - base_idx  # Fractional delay (0 <= mu < 1)

            # Linear interpolation between adjacent samples
            current_sample = (1.0 - mu) * signal[base_idx] + mu * signal[base_idx + 1]

            # Hard decision quantization (Sign-based for generic QAM/PSK)
            decision = np.sign(np.real(current_sample)) + 1j * np.sign(np.imag(current_sample))

            # Mueller & Müller TED
            error = np.real(
                current_sample * np.conj(prev_decision) - prev_sample * np.conj(decision)
            )

            # Loop filter update
            fractional_offset += self.gain * error
            omega = self.sps + fractional_offset

            symbols.append(current_sample)
            strobe_indices.append(base_idx)

            prev_sample = current_sample
            prev_decision = decision

            idx += omega

        return np.array(symbols, dtype=np.complex64), np.array(strobe_indices, dtype=np.int32)
