"""
Ground-Truth Synthetic Signal Generator & Validation Dataset Engine.

Generates deterministic RF signals with exact known ground-truth physics:
- Modulation (BPSK, QPSK, 16-QAM, 2-FSK)
- Controllable AWGN SNR (Clean, 15dB, 10dB, 5dB, 0dB)
- Controllable Carrier Frequency Offset (CFO in Hz)
- Controllable Phase Jitter / Offset
- RRC pulse shaping & symbol clock timing
- Seeded repeatable random bitstreams

Provides:
- Validation dataset generation
- Serialization / deserialization with ground truth metadata
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np


@dataclass
class GroundTruthSignal:
    """Standardized representation of a generated test vector."""
    signal_iq: np.ndarray
    modulation: str
    snr_db: float
    cfo_hz: float
    phase_offset_rad: float
    sample_rate: float
    symbol_rate: float
    num_symbols: int
    seed: int
    payload_bits: np.ndarray
    metadata: Dict[str, Union[str, float, int]] = field(default_factory=dict)


class GroundTruthSignalGenerator:
    """
    Parametric Signal Synthesizer for AMC & Pipeline Validation.
    """

    @staticmethod
    def rrc_pulse_shape(sps: int = 8, alpha: float = 0.35, span: int = 6) -> np.ndarray:
        """Computes Root-Raised-Cosine filter tap coefficients."""
        N = span * sps
        t = np.arange(-N // 2, N // 2 + 1) / sps
        h = np.zeros(len(t))
        for i, val in enumerate(t):
            if np.isclose(val, 0.0):
                h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
            elif np.isclose(np.abs(val), 1.0 / (4.0 * alpha)):
                h[i] = (alpha / np.sqrt(2.0)) * (
                    ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)))
                    + ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha)))
                )
            else:
                num = np.sin(np.pi * val * (1.0 - alpha)) + 4.0 * alpha * val * np.cos(np.pi * val * (1.0 + alpha))
                den = np.pi * val * (1.0 - (4.0 * alpha * val) ** 2)
                h[i] = num / (den + 1e-12)
        return h / np.sqrt(np.sum(h**2))

    @classmethod
    def generate(
        cls,
        modulation: str = "QPSK",
        num_symbols: int = 1024,
        snr_db: float = 20.0,
        cfo_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
        sample_rate: float = 2000000.0,
        symbol_rate: float = 250000.0,
        rrc_alpha: float = 0.35,
        seed: int = 42
    ) -> GroundTruthSignal:
        """
        Generates complex baseband I/Q signal with specific impairments and ground truth.
        """
        rng = np.random.RandomState(seed)
        mod = modulation.upper().replace("-", "")
        sps = max(int(round(sample_rate / symbol_rate)), 1)

        # 1. Generate constellation symbols
        if "BPSK" in mod:
            bits = rng.randint(0, 2, num_symbols)
            syms = 2.0 * bits - 1.0 + 0j
        elif "QPSK" in mod:
            bits = rng.randint(0, 2, num_symbols * 2)
            i_syms = 2.0 * bits[0::2] - 1.0
            q_syms = 2.0 * bits[1::2] - 1.0
            syms = (i_syms + 1j * q_syms) / np.sqrt(2.0)
        elif "16QAM" in mod or "QAM" in mod:
            bits = rng.randint(0, 2, num_symbols * 4)
            b0 = bits[0::4]; b1 = bits[1::4]; b2 = bits[2::4]; b3 = bits[3::4]
            i_syms = (2.0 * b0 - 1.0) * (3.0 - 2.0 * b1)
            q_syms = (2.0 * b2 - 1.0) * (3.0 - 2.0 * b3)
            syms = (i_syms + 1j * q_syms) / np.sqrt(10.0)
        elif "FSK" in mod:
            bits = rng.randint(0, 2, num_symbols)
            # 2-FSK continuous phase frequency modulation
            freq_dev = 50000.0
            freq_series = np.repeat(2.0 * bits - 1.0, sps) * freq_dev
            t = np.arange(len(freq_series)) / sample_rate
            phase = 2.0 * np.pi * np.cumsum(freq_series) / sample_rate
            modulated = np.exp(1j * phase)
            syms = bits + 0j
        else:
            bits = rng.randint(0, 2, num_symbols * 2)
            syms = (2.0 * bits[0::2] - 1.0 + 1j * (2.0 * bits[1::2] - 1.0)) / np.sqrt(2.0)

        # 2. Pulse Shaping (for PSK/QAM)
        if "FSK" not in mod:
            upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
            upsampled[::sps] = syms
            h_rrc = cls.rrc_pulse_shape(sps=sps, alpha=rrc_alpha)
            modulated = np.convolve(upsampled, h_rrc, mode="same")

        # 3. Apply Carrier Frequency Offset (CFO) and Phase Offset
        N = len(modulated)
        t_vec = np.arange(N) / sample_rate
        cfo_rotation = np.exp(1j * (2.0 * np.pi * cfo_hz * t_vec + phase_offset_rad))
        cfo_signal = modulated * cfo_rotation

        # 4. Additive White Gaussian Noise (AWGN)
        sig_pwr = np.mean(np.abs(cfo_signal)**2)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_pwr = sig_pwr / (snr_lin + 1e-12)
        noise = (rng.randn(N) + 1j * rng.randn(N)) * np.sqrt(noise_pwr / 2.0)
        final_signal = cfo_signal + noise

        # Unit Energy Normalization
        final_signal = final_signal / (np.sqrt(np.mean(np.abs(final_signal)**2)) + 1e-12)

        meta = {
            "modulation": modulation,
            "snr_db": float(snr_db),
            "cfo_hz": float(cfo_hz),
            "phase_offset_rad": float(phase_offset_rad),
            "sample_rate": float(sample_rate),
            "symbol_rate": float(symbol_rate),
            "sps": int(sps),
            "num_symbols": int(num_symbols),
            "seed": int(seed)
        }

        return GroundTruthSignal(
            signal_iq=final_signal.astype(np.complex64),
            modulation=modulation,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            phase_offset_rad=phase_offset_rad,
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            num_symbols=num_symbols,
            seed=seed,
            payload_bits=bits,
            metadata=meta
        )

    @classmethod
    def generate_validation_suite(cls, output_dir: Optional[Union[str, Path]] = None) -> List[GroundTruthSignal]:
        """
        Generates standardized validation test dataset across modulations, SNRs, and CFOs.
        """
        mods = ["BPSK", "QPSK", "16-QAM", "2-FSK"]
        snr_levels = [30.0, 15.0, 10.0, 5.0, 0.0]
        cfo_levels = [0.0, 1200.0, -3500.0]

        dataset: List[GroundTruthSignal] = []
        seed_counter = 1000

        for m in mods:
            for s in snr_levels:
                for c in cfo_levels:
                    sig = cls.generate(
                        modulation=m,
                        num_symbols=1024,
                        snr_db=s,
                        cfo_hz=c,
                        phase_offset_rad=0.25 if c != 0 else 0.0,
                        seed=seed_counter
                    )
                    dataset.append(sig)
                    seed_counter += 1

        return dataset
