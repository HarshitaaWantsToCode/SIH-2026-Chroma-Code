"""
M-ary Quadrature Amplitude Modulation (M-QAM) Demodulator.

Provides demodulation and rectangular decision boundaries for QAM-16 and QAM-64.
"""

import numpy as np

from src.dsp.base import BaseDemodulator, DemodulationResult
from src.dsp.synchronization.rrc_filter import RootRaisedCosineFilter


class QAMDemodulator(BaseDemodulator):
    """
    Coherent M-QAM Demodulation Engine.
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        order: int = 16,
        rrc_alpha: float = 0.35
    ) -> None:
        super().__init__(sample_rate, symbol_rate)
        if order not in (16, 64):
            raise ValueError(f"Unsupported QAM order: {order}. Must be 16 or 64.")
        self.order = order
        self.rrc_alpha = rrc_alpha
        self.rrc_filter = RootRaisedCosineFilter()

    def demodulate(self, signal: np.ndarray, **kwargs) -> DemodulationResult:
        # Matched filter
        filtered = self.rrc_filter.apply(signal, self.sps, alpha=self.rrc_alpha)
        # Downsample to symbol strobe (center of eye diagram)
        symbols = filtered[self.sps // 2 :: self.sps]

        # Automatic Gain Control (AGC) scaling for normalized constellation grid
        scale = np.sqrt(np.mean(np.abs(symbols) ** 2))
        normalized_symbols = symbols / (scale if scale > 0 else 1.0)

        bits = self.slice_symbols_to_bits(normalized_symbols)

        return DemodulationResult(
            symbols=normalized_symbols,
            bits=bits,
            sample_rate=self.sample_rate,
            symbol_rate=self.symbol_rate,
            metadata={"modulation": f"{self.order}-QAM", "num_symbols": len(normalized_symbols)}
        )

    def slice_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard decision slicing for Square 16-QAM constellation.
        Levels: [-3, -1, +1, +3] scaled by sqrt(10).
        """
        if self.order == 16:
            norm_factor = np.sqrt(10.0)
            scaled = symbols * norm_factor

            # In-phase decisions
            i = np.real(scaled)
            i_b0 = (i > 0).astype(np.uint8)
            i_b1 = (np.abs(i) < 2).astype(np.uint8)

            # Quadrature decisions
            q = np.imag(scaled)
            q_b0 = (q > 0).astype(np.uint8)
            q_b1 = (np.abs(q) < 2).astype(np.uint8)

            # Interleave 4 bits per symbol
            bitstream = np.empty((len(symbols) * 4,), dtype=np.uint8)
            bitstream[0::4] = i_b0
            bitstream[1::4] = i_b1
            bitstream[2::4] = q_b0
            bitstream[3::4] = q_b1
            return bitstream

        if self.order == 64:
            norm_factor = np.sqrt(42.0)
            scaled = symbols * norm_factor

            def slice_8level(val: np.ndarray):
                # 8 levels: -7, -5, -3, -1, +1, +3, +5, +7 -> 3 bits
                b0 = (val > 0).astype(np.uint8)
                b1 = (np.abs(val) < 4).astype(np.uint8)
                b2 = ((np.abs(val) > 2) & (np.abs(val) < 6)).astype(np.uint8)
                return b0, b1, b2

            i = np.real(scaled)
            q = np.imag(scaled)
            i0, i1, i2 = slice_8level(i)
            q0, q1, q2 = slice_8level(q)

            bitstream = np.empty((len(symbols) * 6,), dtype=np.uint8)
            bitstream[0::6] = i0
            bitstream[1::6] = i1
            bitstream[2::6] = i2
            bitstream[3::6] = q0
            bitstream[4::6] = q1
            bitstream[5::6] = q2
            return bitstream

        # Fallback default 16QAM slicing
        norm_factor = np.sqrt(10.0)
        scaled = symbols * norm_factor
        i = np.real(scaled)
        q = np.imag(scaled)
        bitstream = np.empty((len(symbols) * 4,), dtype=np.uint8)
        bitstream[0::4] = (i > 0).astype(np.uint8)
        bitstream[1::4] = (np.abs(i) < 2).astype(np.uint8)
        bitstream[2::4] = (q > 0).astype(np.uint8)
        bitstream[3::4] = (np.abs(q) < 2).astype(np.uint8)
        return bitstream
