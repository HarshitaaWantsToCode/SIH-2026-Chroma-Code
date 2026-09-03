"""
DSP Base Framework and Demodulator Interfaces.

Defines the abstract contracts and context objects required to execute carrier
recovery, timing recovery, matched filtering, and constellation slicing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class DemodulationResult:
    """
    Standardized payload returned across all demodulation stages.
    """
    symbols: np.ndarray                     # Recovered complex symbol constellation points
    bits: Optional[np.ndarray] = None       # Hard-decision demodulated bits (0/1)
    soft_bits: Optional[np.ndarray] = None   # Log-Likelihood Ratios (LLRs) for soft decoding
    estimated_snr_db: float = 0.0           # Estimated Signal-to-Noise Ratio in dB
    carrier_freq_offset_hz: float = 0.0     # Estimated carrier frequency error Δf
    phase_offset_rad: float = 0.0           # Estimated residual phase θ
    sample_rate: float = 1.0
    symbol_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDemodulator(ABC):
    """
    Abstract Base Class for all digital modulation schemes (PSK, QAM, FSK).
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        samples_per_symbol: Optional[int] = None
    ) -> None:
        """
        Args:
            sample_rate: Sampling frequency Fs (Hz).
            symbol_rate: Baud rate Rs (symbols/sec).
            samples_per_symbol: Oversampling ratio (SPS = Fs / Rs).
        """
        self.sample_rate = float(sample_rate)
        self.symbol_rate = float(symbol_rate)
        raw_sps = samples_per_symbol if samples_per_symbol is not None else (self.sample_rate / max(self.symbol_rate, 1.0))
        self.sps = max(int(round(raw_sps)), 1)

    @abstractmethod
    def demodulate(self, signal: np.ndarray, **kwargs) -> DemodulationResult:
        """
        Executes end-to-end demodulation on the input complex signal.

        Args:
            signal: 1D normalized complex baseband signal.
            **kwargs: Scheme-specific parameters.

        Returns:
            DemodulationResult containing recovered symbols and bitstreams.
        """
        pass

    @abstractmethod
    def slice_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Performs hard-decision maximum-likelihood slicing from complex constellation
        points to binary bit arrays based on Gray code maps.
        """
        pass
