"""
Frequency Shift Keying (FSK) Demodulator.

Implements non-coherent quadrature tone discriminator demodulation for 2-FSK / 4-FSK.
"""

import numpy as np

from src.dsp.base import BaseDemodulator, DemodulationResult


class FSKDemodulator(BaseDemodulator):
    """
    Non-coherent Quadrature Discriminator FSK Demodulator.
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        tone_spacing_hz: float,
        order: int = 2
    ) -> None:
        super().__init__(sample_rate, symbol_rate)
        self.tone_spacing = tone_spacing_hz
        self.order = order

    def demodulate(self, signal: np.ndarray, **kwargs) -> DemodulationResult:
        """
        Instantaneous Frequency Discriminator:
            f[n] = (Fs / (2 * pi)) * angle( s[n] * conj(s[n-1]) )
        """
        # Delay and conjugate product
        d = signal[1:] * np.conj(signal[:-1])
        instantaneous_freq = np.angle(d) * (self.sample_rate / (2.0 * np.pi))

        # Integrate over symbol period
        num_symbols = len(instantaneous_freq) // self.sps
        symbols = np.zeros(num_symbols, dtype=np.float32)

        for k in range(num_symbols):
            segment = instantaneous_freq[k * self.sps : (k + 1) * self.sps]
            symbols[k] = np.mean(segment)

        bits = self.slice_symbols_to_bits(symbols)

        return DemodulationResult(
            symbols=symbols.astype(np.complex64),
            bits=bits,
            sample_rate=self.sample_rate,
            symbol_rate=self.symbol_rate,
            metadata={"modulation": f"{self.order}-FSK", "num_symbols": num_symbols}
        )

    def slice_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Slices discriminator frequency values to binary bit sequence."""
        # For 2-FSK: positive frequency -> 1, negative -> 0
        return (np.real(symbols) > 0).astype(np.uint8)
