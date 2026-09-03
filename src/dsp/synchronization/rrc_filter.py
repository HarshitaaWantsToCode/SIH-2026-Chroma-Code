"""
Root Raised Cosine (RRC) Matched Filter Module.

Implements ideal pulse-shaping and matched filtering for Nyquist inter-symbol
interference (ISI) rejection:
    H_rrc(f) * H_rrc(f) = H_rc(f)
"""

import numpy as np
from scipy.signal import lfilter


class RootRaisedCosineFilter:
    """
    Time-domain Root-Raised-Cosine (RRC) filter synthesis and application.
    """

    @staticmethod
    def design_rrc_taps(
        samples_per_symbol: int,
        alpha: float = 0.35,
        num_symbols: int = 16
    ) -> np.ndarray:
        """
        Generates truncated discrete-time RRC impulse response coefficients.

        Mathematical Formulation:
            h[n] = (sin(pi*t*(1-a)) + 4*a*t*cos(pi*t*(1+a))) / (pi*t*(1 - (4*a*t)^2))
            where t = n / sps

        Args:
            samples_per_symbol: Oversampling factor (sps).
            alpha: Roll-off factor (0.0 <= alpha <= 1.0).
            num_symbols: Filter one-sided symbol span.

        Returns:
            np.ndarray: Normalized filter tap coefficients (Sum(h^2) = 1.0).
        """
        N = num_symbols * samples_per_symbol
        t = np.arange(-N, N + 1, dtype=np.float64) / samples_per_symbol
        h = np.zeros_like(t)

        for idx, val in enumerate(t):
            if np.isclose(val, 0.0):
                h[idx] = 1.0 - alpha + (4.0 * alpha / np.pi)
            elif alpha > 0 and np.isclose(np.abs(val), 1.0 / (4.0 * alpha)):
                term1 = (alpha / np.sqrt(2.0)) * ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)) +
                                                  (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha)))
                h[idx] = term1
            else:
                numerator = np.sin(np.pi * val * (1.0 - alpha)) + 4.0 * alpha * val * np.cos(np.pi * val * (1.0 + alpha))
                denominator = np.pi * val * (1.0 - (4.0 * alpha * val) ** 2)
                h[idx] = numerator / denominator

        # Energy normalization for unity passband gain
        h = h / np.sqrt(np.sum(h ** 2))
        return h

    @classmethod
    def apply(cls, signal: np.ndarray, samples_per_symbol: int, alpha: float = 0.35) -> np.ndarray:
        """Applies matched RRC filtering to complex input signal."""
        taps = cls.design_rrc_taps(samples_per_symbol=samples_per_symbol, alpha=alpha)
        # Apply filter independently to Real and Imaginary components
        filtered_real = lfilter(taps, 1.0, np.real(signal))
        filtered_imag = lfilter(taps, 1.0, np.imag(signal))
        return filtered_real + 1j * filtered_imag
