"""
M-ary Phase Shift Keying (M-PSK) Demodulator.

Implements matched filtering, carrier recovery, timing synchronization, and
constellation mapping for BPSK, QPSK, and 8PSK signals.
"""

import numpy as np

from src.dsp.base import BaseDemodulator, DemodulationResult
from src.dsp.synchronization.costas_loop import CostasLoop
from src.dsp.synchronization.mueller_muller import MuellerMullerTimingRecovery
from src.dsp.synchronization.rrc_filter import RootRaisedCosineFilter


class PSKDemodulator(BaseDemodulator):
    """
    Coherent M-PSK Demodulator pipeline.
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        order: int = 4,
        rrc_alpha: float = 0.35
    ) -> None:
        """
        Args:
            sample_rate: Sampling frequency (Hz).
            symbol_rate: Symbol rate (Baud).
            order: Modulation order M (2 = BPSK, 4 = QPSK, 8 = 8PSK).
            rrc_alpha: Matched filter roll-off factor.
        """
        super().__init__(sample_rate, symbol_rate)
        if order not in (2, 4, 8):
            raise ValueError(f"Unsupported PSK order: {order}. Must be 2, 4, or 8.")

        self.order = order
        self.bits_per_symbol = int(np.log2(order))
        self.rrc_alpha = rrc_alpha

        self.rrc_filter = RootRaisedCosineFilter()
        self.timing_recovery = MuellerMullerTimingRecovery(samples_per_symbol=self.sps)
        self.costas_loop = CostasLoop(order=order)

    def demodulate(self, signal: np.ndarray, **kwargs) -> DemodulationResult:
        """
        Runs full coherent DSP chain:
        1. RRC Matched Filter
        2. Mueller & Müller Clock Recovery
        3. Costas Loop Phase / Frequency Derotation
        4. Symbol Constellation Slicing -> Bits
        """
        # Step 1: Matched Filtering
        filtered = self.rrc_filter.apply(signal, self.sps, alpha=self.rrc_alpha)

        # Step 2: Symbol Timing Synchronization
        sync_symbols, _ = self.timing_recovery.recover_symbols(filtered)

        # Step 3: Carrier Phase and Frequency Tracking
        derotated_symbols, phase_hist, freq_hist = self.costas_loop.process(sync_symbols)

        # Step 4: Symbol-to-bit mapping
        bits = self.slice_symbols_to_bits(derotated_symbols)

        # Estimate SNR via M2M4 second and fourth-order moments
        m2 = np.mean(np.abs(derotated_symbols) ** 2)
        m4 = np.mean(np.abs(derotated_symbols) ** 4)
        snr_est_db = 10.0 * np.log10(max(np.sqrt(max(2 * m2**2 - m4, 1e-12)) / (m2 - np.sqrt(max(2 * m2**2 - m4, 1e-12)) + 1e-12), 1.0))

        return DemodulationResult(
            symbols=derotated_symbols,
            bits=bits,
            estimated_snr_db=float(snr_est_db),
            carrier_freq_offset_hz=float(freq_hist[-1] * (self.symbol_rate / (2 * np.pi))),
            phase_offset_rad=float(phase_hist[-1]),
            sample_rate=self.sample_rate,
            symbol_rate=self.symbol_rate,
            metadata={"modulation": f"{self.order}-PSK", "num_symbols": len(derotated_symbols)}
        )

    def slice_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Quantizes symbols to bits using Gray-coded decision regions.
        """
        if self.order == 2:
            # BPSK: Re(s) > 0 -> 1, Re(s) <= 0 -> 0
            return (np.real(symbols) > 0).astype(np.uint8)

        if self.order == 4:
            # QPSK (Gray coded mapping: 00, 01, 11, 10)
            b0 = (np.real(symbols) > 0).astype(np.uint8)
            b1 = (np.imag(symbols) > 0).astype(np.uint8)
            # Interleave b0 and b1
            bitstream = np.empty((len(symbols) * 2,), dtype=np.uint8)
            bitstream[0::2] = b0
            bitstream[1::2] = b1
            return bitstream

        if self.order == 8:
            # 8-PSK: Partition 8 angular sectors [-pi, pi]
            phases = np.angle(symbols)
            sector = np.mod(np.round(phases / (np.pi / 4)), 8).astype(np.uint8)
            # Gray map: 0->000, 1->001, 2->011, 3->010, 4->110, 5->111, 6->101, 7->100
            gray_map = np.array([
                [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
            ], dtype=np.uint8)
            return gray_map[sector].flatten()

        raise NotImplementedError(f"Slicing for order {self.order} not implemented.")
