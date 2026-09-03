"""
Carrier Phase and Frequency Tracking: Costas Loop.

Implements a 2nd-order Phase-Locked Loop (PLL) tailored for M-PSK suppressed-carrier
synchronization, compensating for carrier frequency offset (CFO) and Doppler shift:
    r[n] = s[n] * exp(-j * theta[n])
"""

from typing import Tuple
import numpy as np


class CostasLoop:
    """
    Discrete-time 2nd-Order Costas Loop for M-ary Phase Shift Keying.
    """

    def __init__(
        self,
        order: int = 2,
        loop_bandwidth: float = 0.01,
        damping_factor: float = 0.707
    ) -> None:
        """
        Args:
            order: Modulation phase ambiguity order (2 for BPSK, 4 for QPSK, 8 for 8PSK).
            loop_bandwidth: Normalized loop filter bandwidth (Bn * Ts).
            damping_factor: Loop damping factor zeta (typically 1/sqrt(2) = 0.707).
        """
        self.order = order
        self.loop_bandwidth = loop_bandwidth
        self.damping_factor = damping_factor

        # Standard Proportional-Integral (PI) 2nd order loop filter coefficients
        denom = (1.0 + 2.0 * damping_factor * loop_bandwidth + loop_bandwidth ** 2)
        self.alpha = (4.0 * damping_factor * loop_bandwidth) / denom  # Proportional gain Kp
        self.beta = (4.0 * loop_bandwidth ** 2) / denom               # Integral gain Ki

    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes carrier recovery iteration over sample array.

        Phase Detector Error Metrics e[n]:
            - BPSK (order=2): e = sign(I) * Q
            - QPSK (order=4): e = sign(I) * Q - sign(Q) * I

        Args:
            signal: 1D complex array at symbol rate or oversampled baseband.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - corrected_signal: Phase/frequency derotated complex samples.
                - phase_history: Estimated instantaneous phase theta[n].
                - freq_history: Estimated instantaneous frequency offset omega[n].
        """
        num_samples = len(signal)
        corrected = np.zeros(num_samples, dtype=np.complex64)
        phase_history = np.zeros(num_samples, dtype=np.float32)
        freq_history = np.zeros(num_samples, dtype=np.float32)

        current_phase = 0.0
        current_freq = 0.0

        for n in range(num_samples):
            # Derotate current sample by phase estimate
            derotated = signal[n] * np.exp(-1j * current_phase)
            corrected[n] = derotated

            i_val = np.real(derotated)
            q_val = np.imag(derotated)

            # Phase Error Detector (PED)
            if self.order == 2:
                # BPSK detector
                error = np.sign(i_val) * q_val
            elif self.order == 4:
                # QPSK detector
                error = np.sign(i_val) * q_val - np.sign(q_val) * i_val
            else:
                # Generalized M-PSK detector: e = imag( s^M )
                error = np.imag(derotated ** self.order)

            # Clamp error to prevent numeric instability
            error = np.clip(error, -2.0, 2.0)

            # Update PI loop filter state
            current_freq += self.beta * error
            current_phase += current_freq + (self.alpha * error)

            # Keep phase bounded within [-pi, pi]
            current_phase = (current_phase + np.pi) % (2.0 * np.pi) - np.pi

            phase_history[n] = current_phase
            freq_history[n] = current_freq

        return corrected, phase_history, freq_history
