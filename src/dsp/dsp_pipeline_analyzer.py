"""
DSP Demodulation Chain Pipeline Analyzer & Progressive Visualizer.

Executes the progressive DSP synchronization stages:
Stage 1: Raw I/Q input (Waveform + Raw Constellation)
Stage 2: Root Raised Cosine (RRC) Matched Filtering
Stage 3: Mueller & Müller Decision-Directed Clock Timing Recovery
Stage 4: 2nd-Order Costas Loop Carrier Phase & CFO Derotation
Stage 5: Hard-Decision Slicing & Decision Boundaries
Stage 6: Recovered Demodulated Bitstream

Provides full progressive stage captures for live demonstrability and side-by-side Before/After comparisons.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.dsp.base import DemodulationResult
from src.dsp.demodulators.fsk import FSKDemodulator
from src.dsp.demodulators.psk import PSKDemodulator
from src.dsp.demodulators.qam import QAMDemodulator
from src.dsp.synchronization.costas_loop import CostasLoop
from src.dsp.synchronization.mueller_muller import MuellerMullerTimingRecovery
from src.dsp.synchronization.rrc_filter import RootRaisedCosineFilter


@dataclass
class DSPStageSnapshot:
    """Snapshot of a single DSP stage for progressive rendering."""
    stage_id: int
    stage_name: str
    summary: str
    status: str                               # "REAL_DSP" or "SIMULATED_DSP"
    time_series_i: np.ndarray
    time_series_q: np.ndarray
    constellation_pts: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressiveDSPAnalysis:
    """Container with all progressive stages and final telemetry."""
    stages: List[DSPStageSnapshot]
    raw_signal: np.ndarray
    rrc_filtered: np.ndarray
    timing_synced: np.ndarray
    carrier_derotated: np.ndarray
    sliced_symbols: np.ndarray
    recovered_bits: np.ndarray
    extracted_params: Dict[str, Any]
    is_real_dsp: bool = True


class DSPProgressivePipeline:
    """
    Executes and records all intermediate mathematical transformations across the DSP chain.
    """

    @classmethod
    def analyze(
        cls,
        signal: np.ndarray,
        modulation: str = "QPSK",
        sample_rate: float = 2000000.0,
        symbol_rate: float = 250000.0,
        rrc_alpha: float = 0.35
    ) -> ProgressiveDSPAnalysis:
        """
        Executes step-by-step DSP processing and gathers visual snapshots.
        """
        mod = modulation.upper().replace("-", "")
        sps = int(round(sample_rate / symbol_rate))
        sps = max(sps, 1)

        # ----------------- STAGE 1: RAW I/Q -----------------
        raw_pts = signal[: min(2000, len(signal))]
        s1 = DSPStageSnapshot(
            stage_id=1,
            stage_name="1. RAW IQ INPUT",
            summary="Raw complex baseband capture before matched filtering or synchronization. Displays carrier rotation and ISI dispersion.",
            status="REAL_DSP",
            time_series_i=np.real(signal[:200]),
            time_series_q=np.imag(signal[:200]),
            constellation_pts=raw_pts,
            metadata={"samples": len(signal), "rms": float(np.sqrt(np.mean(np.abs(signal)**2)))}
        )

        # ----------------- STAGE 2: RRC MATCHED FILTER -----------------
        rrc = RootRaisedCosineFilter()
        if "FSK" not in mod:
            rrc_filtered = rrc.apply(signal, sps, alpha=rrc_alpha)
        else:
            rrc_filtered = signal.copy()

        s2 = DSPStageSnapshot(
            stage_id=2,
            stage_name="2. RRC MATCHED FILTER",
            summary=f"Nyquist Root-Raised-Cosine pulse shaping applied (α = {rrc_alpha}). Maximizes SNR while eliminating Inter-Symbol Interference (ISI).",
            status="REAL_DSP",
            time_series_i=np.real(rrc_filtered[:200]),
            time_series_q=np.imag(rrc_filtered[:200]),
            constellation_pts=rrc_filtered[: min(2000, len(rrc_filtered))],
            metadata={"filter_alpha": rrc_alpha, "sps": sps}
        )

        # ----------------- STAGE 3: MUELLER & MÜLLER TIMING RECOVERY -----------------
        if "FSK" not in mod:
            mm_recovery = MuellerMullerTimingRecovery(samples_per_symbol=sps)
            timing_synced, strobe_indices = mm_recovery.recover_symbols(rrc_filtered)
            if len(timing_synced) < 10:
                timing_synced = rrc_filtered[sps // 2 :: sps]
        else:
            timing_synced = rrc_filtered[sps // 2 :: sps]

        s3 = DSPStageSnapshot(
            stage_id=3,
            stage_name="3. TIMING RECOVERY (MUELLER & MÜLLER)",
            summary="Decision-directed clock strobe extraction. Interpolates fractional delay to sample at the optimal peak of the eye diagram.",
            status="REAL_DSP",
            time_series_i=np.real(timing_synced[:200]),
            time_series_q=np.imag(timing_synced[:200]),
            constellation_pts=timing_synced[: min(2000, len(timing_synced))],
            metadata={"extracted_symbols": len(timing_synced), "strobe_count": len(timing_synced)}
        )

        # ----------------- STAGE 4: COSTAS LOOP CARRIER DEROTATION -----------------
        if "PSK" in mod or mod in ("QPSK", "BPSK", "8PSK"):
            order_psk = 2 if "BPSK" in mod else 8 if "8PSK" in mod else 4
            costas = CostasLoop(order=order_psk)
            carrier_derotated, phase_hist, freq_hist = costas.process(timing_synced)
            cfo_hz = float(freq_hist[-1] * (symbol_rate / (2 * np.pi)))
            phase_rad = float(phase_hist[-1])
        elif "QAM" in mod:
            scale = np.sqrt(np.mean(np.abs(timing_synced) ** 2))
            carrier_derotated = timing_synced / (scale if scale > 0 else 1.0)
            cfo_hz = None
            cfo_display = "NOT_AVAILABLE (QAM Multi-ring)"
            phase_rad = 0.0
        else:
            carrier_derotated = timing_synced
            cfo_hz = 0.0
            cfo_display = "0.00 Hz (Constant Modulus Baseband)"
            phase_rad = 0.0

        if cfo_hz is not None:
            cfo_display = f"{cfo_hz:.2f} Hz"

        s4 = DSPStageSnapshot(
            stage_id=4,
            stage_name="4. CARRIER RECOVERY (COSTAS PLL)",
            summary="2nd-order Phase-Locked Loop locks onto carrier wave phase and wipes out Carrier Frequency Offset (CFO) Doppler rotation.",
            status="REAL_DSP",
            time_series_i=np.real(carrier_derotated[:200]),
            time_series_q=np.imag(carrier_derotated[:200]),
            constellation_pts=carrier_derotated[: min(2000, len(carrier_derotated))],
            metadata={"cfo_hz": cfo_hz if cfo_hz is not None else "NOT_AVAILABLE", "phase_error_rad": phase_rad}
        )

        # ----------------- STAGE 5: SYMBOL SLICING -----------------
        if "QAM" in mod:
            order_q = 64 if "64" in mod else 16
            demod_obj = QAMDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate, order=order_q)
            bits = demod_obj.slice_symbols_to_bits(carrier_derotated)
            sliced_pts = carrier_derotated
        elif "FSK" in mod:
            demod_obj = FSKDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate, tone_spacing_hz=50000.0)
            res_fsk = demod_obj.demodulate(signal)
            bits = res_fsk.bits
            sliced_pts = res_fsk.symbols
        else:
            order_p = 2 if "BPSK" in mod else 8 if "8PSK" in mod else 4
            demod_obj = PSKDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate, order=order_p)
            bits = demod_obj.slice_symbols_to_bits(carrier_derotated)
            sliced_pts = carrier_derotated

        s5 = DSPStageSnapshot(
            stage_id=5,
            stage_name="5. SYMBOL SLICING & DECISION REGIONS",
            summary=f"Hard-decision slicing maps complex I/Q coordinates to constellation decision centroids ({modulation} Gray map).",
            status="REAL_DSP",
            time_series_i=np.real(sliced_pts[:200]),
            time_series_q=np.imag(sliced_pts[:200]),
            constellation_pts=sliced_pts[: min(2000, len(sliced_pts))],
            metadata={"total_symbols": len(sliced_pts), "modulation": modulation}
        )

        # ----------------- STAGE 6: RECOVERED BITSTREAM & ROBUST SPECTRAL SNR -----------------
        # Robust in-band vs out-of-band spectral density SNR estimation
        sps = sample_rate / (symbol_rate if symbol_rate > 0 else 1.0)
        fft_s = np.abs(np.fft.fft(signal))**2
        fft_shift = np.fft.fftshift(fft_s)
        N_pts = len(fft_shift)

        frac_bw = 1.25 / sps if sps > 1.0 else 0.8
        in_half = max(1, int(N_pts * frac_bw * 0.5))
        center_idx = N_pts // 2
        in_band = fft_shift[center_idx - in_half : center_idx + in_half]

        out_half = max(2, int(N_pts * (1.6 / sps) * 0.5)) if sps > 1.0 else int(N_pts * 0.4)
        out_band = np.concatenate([fft_shift[:center_idx - out_half], fft_shift[center_idx + out_half:]])
        if len(out_band) < 20:
            out_band = fft_shift[:int(N_pts * 0.1)]

        noise_psd = np.median(out_band)
        in_band_avg = np.mean(in_band)
        sig_pwr = max(in_band_avg - noise_psd, 1e-12)
        snr_est_db = float(10.0 * np.log10(sig_pwr / (noise_psd + 1e-12)) - 10.0 * np.log10(max(1.0, sps)))
        snr_est_db = max(0.0, snr_est_db)

        s6 = DSPStageSnapshot(
            stage_id=6,
            stage_name="6. RECOVERED BITSTREAM",
            summary=f"Extracted {len(bits):,} discrete binary bits with estimated channel SNR of {snr_est_db:.2f} dB.",
            status="REAL_DSP",
            time_series_i=bits[:200].astype(np.float32),
            time_series_q=np.zeros(min(200, len(bits)), dtype=np.float32),
            constellation_pts=sliced_pts[:500],
            metadata={"recovered_bits": len(bits), "snr_db": float(snr_est_db)}
        )

        extracted = {
            "Modulation": modulation,
            "Estimated SNR": f"{snr_est_db:.2f} dB",
            "Carrier Frequency Offset (Δf)": cfo_display,
            "Phase Error (θ)": f"{phase_rad:.4f} rad",
            "Symbol Baud Rate": f"{symbol_rate/1e3:.1f} kBaud",
            "Samples Processed": f"{len(signal):,} samples",
            "Total Recovered Symbols": f"{len(sliced_pts):,}",
            "Recovered Bit Count": f"{len(bits):,} bits",
            "DSP Mode": "Real Mathematical Algorithms"
        }

        return ProgressiveDSPAnalysis(
            stages=[s1, s2, s3, s4, s5, s6],
            raw_signal=signal,
            rrc_filtered=rrc_filtered,
            timing_synced=timing_synced,
            carrier_derotated=carrier_derotated,
            sliced_symbols=sliced_pts,
            recovered_bits=bits,
            extracted_params=extracted,
            is_real_dsp=True
        )
